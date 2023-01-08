#include "QIEFPDomainService.h"

#include <QIMLCommon.h>
#include <AlgorithmEFPCpp.h>
#include <AlgorithmEFPCuda.h>

#include <CurveAvgCalculation.h>
#include <PythonCommon.h>

#include <boost/format.hpp>

#include <string>
#include <chrono>
#include <future>
#include <thread>
#include <array>
#include <iterator>
#include <vector>

using namespace BoreholeImagingStudio;
using namespace BoreholeImagingStudio::QIML;

namespace
{
	int efpcompute_progress_wait_seconds = 60; //Currently its set to 60 seconds,

	std::string video_card_error = "Computation on GPU will NOT run in the current machine. QI CUDA algorithms for SMC and EFP\n"
								   "are not supported on machines without NVIDIA video cards with certain properties.\n"
								   "The application failed to load CUDAAlgoritms.dll/libCUDAAlgoritms.so .\n\nPlease run the application using CPU.\n\n"
								   "Please contact the BIS technical team (Petrosite.Support@Halliburton.com) if you need more information.";	

}

QIEFPDomainService::QIEFPDomainService(CurvesRepository &&curves_repository, ToolRepository &&tool_repository)
	: m_tool_repository(std::move(tool_repository)),
	  m_curves_repository(std::move(curves_repository))
{
	
}

std::pair<ComputeStatus, std::string> QIEFPDomainService::Init(const Common::DBLocation &db_location, std::vector<std::string> &freqs)
{
	LOG_INFO(LogLayer::Domain) << "QIEFPDomainService Init- Started ";
	m_cancel_requested = false;
	m_process_completed = false;
	#ifdef __linux
		PythonCommon::InitPythonWithoutCallbacks();
	#endif

	try
	{
		m_available_freqs.clear();		
		for (auto &freq : freqs)
		{
			m_available_freqs.push_back(freq);
		}
		m_tool_info = m_tool_repository.GetToolInfo(db_location.record_name, freqs);	
	}
	catch (const std::exception &ex)
	{
		LOG_ERROR(LogLayer::Domain) << ex.what();
		std::string msg = "Failed to create ToolCore object! " + std::string(ex.what());
		LOG_ERROR(LogLayer::Domain) << msg;
		return {ComputeStatus::FAILED, msg };
	}

	LOG_INFO(LogLayer::Domain) << "QIEFPDomainService Init- Completed ";
	return {ComputeStatus::IN_PROGRESS, ""};
}

std::string QIEFPDomainService::GetToolType()
{
	return m_tool_info.tool_type;
}

void QIEFPDomainService::SetCancellationRequest()
{
	m_cancel_requested = true;
}

std::pair<ComputeStatus, std::string> QIEFPDomainService::Compute(
	const Common::DBLocation &db_location,
	const double min_depth,
	const double max_depth,
	BoreholeImagingStudio::QIML::EFPParams &inputParams,
	bool IsComputeChunk,
	std::function<void(ComputeStatus, std::string, float)> progress_cb)
{
	m_cancel_requested = false;
	if (m_cancel_requested)
	{
		progress_cb(ComputeStatus::CANCELLED, "Process Cancelled.", 100);
		return {ComputeStatus::CANCELLED, "Process Cancelled."};
	}

	boost::format range_formatter("%1$.2f to %2$.2f");
	std::string str_depth_range = (range_formatter % min_depth % max_depth).str();
	LOG_INFO(LogLayer::Domain) << "Compute started for Depth Range " + str_depth_range;
	m_input_db_location = db_location;
	m_output_db_location = db_location;

	m_min_depth = min_depth;
	m_max_depth = max_depth;

	//If QI EFP is in GPU, check NVidia card and driver support here
	if (!inputParams.is_cpu)
	{
		LOG_INFO(LogLayer::Domain) << "EFP GPU run requested.. ";
		std::string device_name, drv_check;
		const auto is_supported = AlgorithmEFPCuda::CheckIfCuda110IsSupported(device_name, drv_check);
		if (!is_supported)
		{
			LOG_INFO(LogLayer::Domain) << "GPU video card error-> " << video_card_error;
			return {ComputeStatus::FAILED, video_card_error};
		}
	}

	auto begin_proc = std::chrono::high_resolution_clock::now();

	auto dataprep_result = DataPrepEFP(inputParams);
	if (!dataprep_result.first.empty())
	{
		if (dataprep_result.first == "CANCELLED" || m_cancel_requested)
		{
			progress_cb(ComputeStatus::CANCELLED, "Process Cancelled.", 100);
			return {ComputeStatus::CANCELLED, "Process Cancelled."};
		}

		std::string msg = "DataPrep Failed! " + dataprep_result.first;
		LOG_ERROR(LogLayer::Domain) << msg;
		progress_cb(ComputeStatus::FAILED, msg, 100);
		return {ComputeStatus::FAILED, msg};
	}
	if (!IsComputeChunk) //ComputeChunk=false means, its called by Processing framework to run multiple instances of this services in cloud, then disable multithreading that uses all cores of node. 
	{
		dataprep_result.second.get()->use_multiprocessing = false;
	}

	auto msg = "DataPrep for Depth Range " + str_depth_range;
	progress_cb(ComputeStatus::IN_PROGRESS, msg, m_pb_dataprep_percentage);

	auto end_proc = std::chrono::high_resolution_clock::now();
	double time_taken_secs = std::chrono::duration<double>(end_proc - begin_proc).count();

	LOG_INFO(LogLayer::Domain) << "DataPrep Completed and process took " << time_taken_secs;
	
	begin_proc = std::chrono::high_resolution_clock::now();
	std::future<OutputDataEFP> efpStatus;
	try
	{
		auto inputData = *(dataprep_result.second.get());
		msg = "Compute in Progress for depths " + str_depth_range;
		progress_cb(ComputeStatus::IN_PROGRESS, msg, 0.0);
		if (m_cancel_requested)
		{
			progress_cb(ComputeStatus::CANCELLED, "Process Cancelled.", 100);
			return {ComputeStatus::CANCELLED, "Process Cancelled."};
		}

		std::future_status compute_status = std::future_status::deferred;
		efpStatus = std::move(std::async(std::launch::async, &QIEFPDomainService::ComputeEFPAsync, this, std::ref(inputData), inputParams.is_cpu));
		do
		{
			compute_status = efpStatus.wait_for(std::chrono::seconds(efpcompute_progress_wait_seconds));
			progress_cb(ComputeStatus::IN_PROGRESS, msg, 0.0);
		} while (compute_status != std::future_status::ready);
	}
	catch (const std::exception &e)
	{
		std::string msg = "ComputeEFP Failed!" + std::string(e.what());
		LOG_ERROR(LogLayer::Domain) << msg;
		progress_cb(ComputeStatus::FAILED, msg, 100);
		return {ComputeStatus::FAILED, msg};
	}
	catch (...)
	{
		std::string msg = "ComputeEFP Failed! for depth range " + str_depth_range;
		LOG_ERROR(LogLayer::Domain) << msg;
		progress_cb(ComputeStatus::FAILED, msg, 100);
		return {ComputeStatus::FAILED, msg};
	}

	if (m_cancel_requested)
	{
		progress_cb(ComputeStatus::CANCELLED, "Process Cancelled.", 100);
		return {ComputeStatus::CANCELLED, "Process Cancelled."};
	}

	msg = "Computation completed for depth range " + str_depth_range;
	progress_cb(ComputeStatus::IN_PROGRESS, msg, m_pb_compute_percentage);

	end_proc = std::chrono::high_resolution_clock::now();
	time_taken_secs = std::chrono::duration<double>(end_proc - begin_proc).count();
	LOG_INFO(LogLayer::Domain) << "Compute completed in " << time_taken_secs << " seconds";

	auto efp_out = std::move(efpStatus.get());

	if (m_cancel_requested)
	{
		progress_cb(ComputeStatus::CANCELLED, "Process Cancelled.", 100);
		return {ComputeStatus::CANCELLED, "Process Cancelled."};
	}

	FreqSort(efp_out.rf_est, m_smc_data.oper_freqs, m_orig_freqs);
	FreqSort(efp_out.epsrf_est, m_smc_data.oper_freqs, m_orig_freqs);
	FreqSort(efp_out.Zreconst_abs, m_smc_data.oper_freqs, m_orig_freqs);
	FreqSort(efp_out.Zreconst_phase, m_smc_data.oper_freqs, m_orig_freqs);
	FreqSort(efp_out.Zreconst_real, m_smc_data.oper_freqs, m_orig_freqs);
	efp_out.oper_freqs = m_orig_freqs;

	m_output_db_location.desc_name = inputParams.output_desc_name;

	msg = "Writing Curves to Insite for depths " + str_depth_range;
	progress_cb(ComputeStatus::IN_PROGRESS, msg, 0.0);

	auto saveresult = SaveEFP(efp_out);
	if (!saveresult.empty())
	{
		if (saveresult == "CANCELLED" || m_cancel_requested)
		{
			progress_cb(ComputeStatus::CANCELLED, "Process Cancelled.", 100);
			return {ComputeStatus::CANCELLED, "Process Cancelled."};
		}

		std::string msg = "Writing curves to Insite Failed! " + saveresult;
		LOG_ERROR(LogLayer::Domain) << msg;
		progress_cb(ComputeStatus::FAILED, msg, 100);
		return {ComputeStatus::FAILED, msg };
	}

	msg = "Writing Curves to Insite completed. " + str_depth_range;
	//Compute called by ComputeByChunks, compute status will be in progress, compute_status=SUCCESS will be set in computebychunks method
	if (IsComputeChunk)
	{
		progress_cb(ComputeStatus::IN_PROGRESS, msg, m_pb_writing_percentage);
	}
	else //Compute called directly for single chunk depth range, so compute status success
	{
		progress_cb(ComputeStatus::SUCCESS, msg, m_pb_writing_percentage);
	}
	
	return {ComputeStatus::SUCCESS, msg};
}

std::pair<ComputeStatus, std::string> QIEFPDomainService::ComputeByChunk(const Common::DBLocation &db_location,
																		 const double min_depth,
																		 const double max_depth,
																		 QIML::EFPParams &inputParams,
																		 std::function<void(ComputeStatus, std::string, float)> progress_cb)
{

	if (m_cancel_requested)
	{
		progress_cb(ComputeStatus::CANCELLED, "Process Cancelled.", 100);
		return {ComputeStatus::CANCELLED, "Process Cancelled."};
	}

	try
	{
		std::vector<std::string> curves;		
		const auto pad_curve_names = m_tool_info.pad_curve_names[0]; //read first frequency pad curve names to get depth events.
		LOG_INFO(LogLayer::Domain) << "Pad Curve name to read depth values to do depth chunking " << pad_curve_names.at(0);
		curves.push_back(pad_curve_names.at(0)); //DataAccessService pbSetDatasetColumnMajor doesnt support reading only depth curve, using this work around to read depth events for chunking
		auto db_loc_re = db_location;
		db_loc_re.record_name = m_tool_info.smc_re_record;
		auto curve_result = m_curves_repository.ReadCurves(db_loc_re, curves, min_depth, max_depth);
		if (m_cancel_requested)
		{
			progress_cb(ComputeStatus::CANCELLED, "Process Cancelled.", 100);
			return {ComputeStatus::CANCELLED, "Process Cancelled."};
		}	
		
		const auto &data_re = curve_result[0];
		const long efp_top_ind = std::lower_bound(data_re->events, data_re->events + data_re->numberOfEvents, min_depth) - data_re->events;
		const long efp_bot_ind = std::upper_bound(data_re->events, data_re->events + data_re->numberOfEvents, max_depth) - data_re->events - 1;
		std::vector<std::pair<double, double>> chunks;

		auto chunk_size = inputParams.chunk_size;
		const auto total_number_of_events = efp_bot_ind - efp_top_ind + 1;
		auto top_idx = efp_top_ind;
		auto bot_idx = std::min(top_idx + chunk_size, efp_bot_ind);

		const auto number_of_chunks = ceilf(static_cast<float>(total_number_of_events) / chunk_size);
		LOG_INFO(LogLayer::Domain) << "Number of chunks are " << number_of_chunks;
		const int pb_increment = 100 / number_of_chunks;
		m_pb_compute_percentage = pb_increment * 0.8;
		m_pb_dataprep_percentage = pb_increment * 0.1;
		m_pb_writing_percentage = pb_increment * 0.1;
		
		while (top_idx <= efp_bot_ind)
		{
			auto chunk_min_Depth = data_re->events[top_idx];
			auto chunk_max_Depth = data_re->events[bot_idx];

			chunks.push_back(std::make_pair(chunk_min_Depth, chunk_max_Depth));
			top_idx = bot_idx + 1;
			bot_idx = std::min(top_idx + chunk_size, efp_bot_ind);
		}
		auto progess = 0;
		for (const auto &curve_chunk : chunks)
		{
			if (m_cancel_requested)
			{
				progress_cb(ComputeStatus::CANCELLED, "Process Cancelled.", 100);
				return {ComputeStatus::CANCELLED, "Process Cancelled."};
			}

			auto result = Compute(db_location, curve_chunk.first, curve_chunk.second, inputParams, true, progress_cb);
			if (result.first == ComputeStatus::CANCELLED || m_cancel_requested)
			{
				progress_cb(ComputeStatus::CANCELLED, "Process Cancelled.", 100);
				return {ComputeStatus::CANCELLED, "Process Cancelled."};
			}
			else if (result.first == ComputeStatus::FAILED)
			{
				progress_cb(ComputeStatus::FAILED, result.second, 100.0);
				return {ComputeStatus::FAILED, result.second};
			}

			LOG_INFO(LogLayer::Domain) << "Compute Progress percent " << std::to_string(pb_increment);
			progess = pb_increment;
		}
	}
	catch (const std::exception &ex)
	{
		std::string msg = std::string(ex.what()) + " in QIEFP ComputeByChunks";
		LOG_ERROR(LogLayer::Domain) << msg;
		progress_cb(ComputeStatus::FAILED, msg, 100.0);
		return {ComputeStatus::FAILED, msg};
	}

	if (m_cancel_requested)
	{
		progress_cb(ComputeStatus::CANCELLED, "Process Cancelled.", 100);
		return {ComputeStatus::CANCELLED, "Process Cancelled."};
	}

	progress_cb(ComputeStatus::SUCCESS, "Compute by Chunk completed", 100.0);

	return {ComputeStatus::SUCCESS, ""};
}

std::pair<std::string, std::unique_ptr<InputDataEFP>> QIEFPDomainService::DataPrepEFP(QIML::EFPParams &inputParams)
{
	auto status = ReadSMCData();
	
	if (status.first == ComputeStatus::CANCELLED || m_cancel_requested)
	{
		return {"CANCELLED", nullptr};
	}
	else if (status.first == ComputeStatus::FAILED)
	{
		LOG_ERROR(LogLayer::Domain) << "Reading SMC data failed " << status.second;
		return { status.second, nullptr };
	}

	auto validation_res = m_smc_data.IsValid();

	if (!validation_res.empty())
	{
		LOG_ERROR(LogLayer::Domain) << validation_res;
		return {validation_res, nullptr};
	}

	if (inputParams.use_ig)
	{		
		LOG_INFO(LogLayer::Domain) << "Read SMC Inital Guess data. ";
		auto smcigStatus = ReadSMCIG();
		if (smcigStatus.first == ComputeStatus::CANCELLED || m_cancel_requested)
		{
			return { "CANCELLED", nullptr };
		}
		else if (smcigStatus.first == ComputeStatus::FAILED)
		{
			LOG_ERROR(LogLayer::Domain) << "Reading SMC Initial Guess data failed " << smcigStatus.second;
			return { smcigStatus.second, nullptr };
		}			
	}

	LOG_INFO(LogLayer::Domain) << "SMC Inital Guess data loading completed. ";

	m_orig_freqs = m_smc_data.oper_freqs;
	auto sorted_freq = m_smc_data.oper_freqs;
	sort(sorted_freq.begin(), sorted_freq.end());
	if (m_cancel_requested)
	{
		return {"CANCELLED", nullptr};
	}

	FreqSort(m_smc_data.comp_cal_freq_logs_re, m_smc_data.oper_freqs, sorted_freq);
	FreqSort(m_smc_data.comp_cal_freq_logs_im, m_smc_data.oper_freqs, sorted_freq);
	FreqSort(m_smc_data.cal_freq_logs_re, m_smc_data.oper_freqs, sorted_freq);
	FreqSort(m_smc_data.mangle_use, m_smc_data.oper_freqs, sorted_freq);
	FreqSort(m_smc_data.epsrm_use, m_smc_data.oper_freqs, sorted_freq);

	auto sortedWts = inputParams.wts;
	FreqSort(sortedWts, m_smc_data.oper_freqs, sorted_freq, sortedWts.size() / sorted_freq.size());

	m_smc_data.oper_freqs = sorted_freq;	

	std::vector<double> curr_epsrm_use;
	std::vector<double> curr_mangle_use;

	for (const auto &curve : m_smc_data.epsrm_use)
		curr_epsrm_use.push_back(curve->data[0]);

	for (const auto &curve : m_smc_data.mangle_use)
		curr_mangle_use.push_back(curve->data[0]);
	if (m_cancel_requested)
	{
		return {"CANCELLED", nullptr};
	}

	m_top_ind = 0;
	m_bot_ind = m_smc_data.comp_cal_freq_logs_re[0]->numberOfEvents - 1;
	std::unique_ptr<InputDataEFP> inputData = std::make_unique<InputDataEFP>(m_smc_data.comp_cal_freq_logs_re, m_smc_data.comp_cal_freq_logs_im,					
																m_smc_data.cal_freq_logs_re, sorted_freq, curr_epsrm_use, curr_mangle_use,
																inputParams.model_params_full_path, inputParams.reg_param, sortedWts, m_top_ind,
															m_bot_ind, m_tool_info.tool_type);	
	if (inputParams.use_ig)
	{
		SetSMCInitGuess(m_smc_data.init_guess.at(m_tool_info.smc_ig_rxo_record).get(), m_smc_data.init_guess.at(m_tool_info.smc_ig_epsrf_record).get(),
			inputData.get());		
	}

	return {"", std::move(inputData)};
}

OutputDataEFP QIEFPDomainService::ComputeEFPAsync(InputDataEFP &inputDataEFP, bool isCPU)
{
	const auto begin_proc = std::chrono::high_resolution_clock::now();

	auto auto_res = isCPU ? AlgorithmEFPCpp().Compute(inputDataEFP)
						  : AlgorithmEFPCuda().Compute(inputDataEFP);

	const auto end_proc = std::chrono::high_resolution_clock::now();
	double time_taken_secs = std::chrono::duration<double>(end_proc - begin_proc).count();

	LOG_INFO(LogLayer::Domain) << "Efp algorithm Completed and process took " << time_taken_secs;

	const auto &data_re = m_smc_data.comp_cal_freq_logs_re[0];
	OutputDataEFP efp_out(data_re->events,
						  m_top_ind, m_bot_ind,
						  m_smc_data.cal_freq_logs_re[0]->numberOfDataSamples,
						  static_cast<int>(m_smc_data.oper_freqs.size()),
						  m_smc_data.oper_freqs);

	efp_out.Append(auto_res.events,
				   auto_res.total_num_buttons,
				   auto_res.num_freq,
				   auto_res.standoff_est.data(),
				   auto_res.rf_est.data(),
				   auto_res.epsrf_est.data(),
				   auto_res.misfitall.data(), auto_res.Zreconst.data());

	return efp_out;
}

OutputDataEFP QIEFPDomainService::ComputeEFP(InputDataEFP &inputDataEFP,
											 bool isCPU)
{
	if (isCPU)
	{
		LOG_INFO(LogLayer::Domain) << "CPU Efp algorithm started.";
	}
	else
	{

		LOG_INFO(LogLayer::Domain) << "GPU CUDA Efp algorithm started.";
	}

	const auto begin_proc = std::chrono::high_resolution_clock::now();

	auto auto_res = isCPU ? AlgorithmEFPCpp().Compute(inputDataEFP)
						  : AlgorithmEFPCuda().Compute(inputDataEFP);

	const auto end_proc = std::chrono::high_resolution_clock::now();
	auto time_taken = std::chrono::duration<double>(end_proc - begin_proc).count();

	LOG_INFO(LogLayer::Domain) << "Efp algorithm Completed and process took " << time_taken;

	const auto &data_re = m_smc_data.comp_cal_freq_logs_re[0];
	OutputDataEFP efp_out(data_re->events,
						  m_top_ind, m_bot_ind,
						  m_smc_data.cal_freq_logs_re[0]->numberOfDataSamples,
						  static_cast<int>(m_smc_data.oper_freqs.size()),
						  m_smc_data.oper_freqs);

	efp_out.Append(auto_res.events,
				   auto_res.total_num_buttons,
				   auto_res.num_freq,
				   auto_res.standoff_est.data(),
				   auto_res.rf_est.data(),
				   auto_res.epsrf_est.data(),
				   auto_res.misfitall.data(),
				   auto_res.Zreconst.data());
	LOG_INFO(LogLayer::Domain) << "CPU Efp algorithm result re-arranged for Insite writing.";
	return efp_out;
}

std::string QIEFPDomainService::SaveEFP(OutputDataEFP &efp_out)
{
	if (m_cancel_requested)
	{
		return "CANCELLED";
	}

	LOG_INFO(LogLayer::Domain) << "Writing Curves to Insite DB started.";

	const auto begin_proc = std::chrono::high_resolution_clock::now();

	const auto events = efp_out.rf_est[0]->events;
	const auto num_events = efp_out.rf_est[0]->numberOfEvents;
	const auto num_data_samples = efp_out.rf_est[0]->numberOfDataSamples;
	const auto num_freqs = static_cast<int>(efp_out.rf_est.size() - 1);
	const auto num_pads = m_tool_info.number_of_pads;
	const auto num_buttons = m_tool_info.number_of_buttons;

	std::vector<std::unique_ptr<Common::logCurve>> logs_rf;
	logs_rf.reserve(num_freqs * (num_pads + 1));

	std::vector<std::unique_ptr<Common::logCurve>> logs_epsrf;
	logs_epsrf.reserve(num_freqs * (num_pads + 1));

	std::vector<std::unique_ptr<Common::logCurve>> logs_abs;
	logs_abs.reserve(num_freqs * (num_pads + 1));

	std::vector<std::unique_ptr<Common::logCurve>> logs_phase;
	logs_phase.reserve(num_freqs * (num_pads + 1));

	std::vector<std::unique_ptr<Common::logCurve>> logs_re;
	logs_re.reserve(num_freqs * (num_pads + 1));

	for (int freq_i = 0; freq_i < m_available_freqs.size(); freq_i++)
	{
		if (m_cancel_requested)
		{
			return "CANCELLED";
		}

		const auto pad_curve_names = m_tool_info.pad_curve_names[freq_i];

		for (const auto &pad_name : pad_curve_names)
		{
			logs_rf.push_back(std::make_unique<Common::logCurve>(pad_name, events, num_events, num_buttons));
			logs_epsrf.push_back(std::make_unique<Common::logCurve>(pad_name, events, num_events, num_buttons));
			logs_abs.push_back(std::make_unique<Common::logCurve>(pad_name, events, num_events, num_buttons));
			logs_re.push_back(std::make_unique<Common::logCurve>(pad_name, events, num_events, num_buttons));
		}

		const auto phase_curve_names = m_tool_info.phase_curve_names[freq_i];

		for (const auto &phase_name : phase_curve_names)
			logs_phase.push_back(std::make_unique<Common::logCurve>(phase_name, events, num_events, num_buttons));
	}

	for (auto event_i = 0, event_start = 0, pad_event_start = 0;
		 event_i < num_events; ++event_i, event_start += num_data_samples, pad_event_start += num_buttons)
	{
		for (auto freq_i = 0, freq_start = 0; freq_i < num_freqs; ++freq_i, freq_start += num_pads)
		{
			for (auto pad_i = 0, pad_start = 0; pad_i < num_pads; ++pad_i, pad_start += num_buttons)
			{
				if (m_cancel_requested)
				{
					return "CANCELLED";
				}

				std::copy(efp_out.rf_est[freq_i]->DataBegin() + event_start + pad_start,
					 efp_out.rf_est[freq_i]->DataBegin() + event_start + pad_start + num_buttons,
					 logs_rf[freq_start + pad_i]->DataBegin() + pad_event_start);

				std::copy(efp_out.epsrf_est[freq_i]->DataBegin() + event_start + pad_start,
					 efp_out.epsrf_est[freq_i]->DataBegin() + event_start + pad_start + num_buttons,
					 logs_epsrf[freq_start + pad_i]->DataBegin() + pad_event_start);

				std::copy(efp_out.Zreconst_abs[freq_i]->DataBegin() + event_start + pad_start,
					 efp_out.Zreconst_abs[freq_i]->DataBegin() + event_start + pad_start + num_buttons,
					 logs_abs[freq_start + pad_i]->DataBegin() + pad_event_start);

				std::copy(efp_out.Zreconst_phase[freq_i]->DataBegin() + event_start + pad_start,
					 efp_out.Zreconst_phase[freq_i]->DataBegin() + event_start + pad_start + num_buttons,
					 logs_phase[freq_start + pad_i]->DataBegin() + pad_event_start);

				std::copy(efp_out.Zreconst_real[freq_i]->DataBegin() + event_start + pad_start,
					 efp_out.Zreconst_real[freq_i]->DataBegin() + event_start + pad_start + num_buttons,
					 logs_re[freq_start + pad_i]->DataBegin() + pad_event_start);
			}
		}
	}

	for (int freq_i = 0; freq_i < m_available_freqs.size(); freq_i++)
	{
		if (m_cancel_requested)
		{
			return "CANCELLED";
		}

		const auto freq_curve_name = m_tool_info.frequency_curve_names[freq_i];

		logs_rf.push_back(std::make_unique<Common::logCurve>(freq_curve_name, events, num_events, 1, efp_out.oper_freqs[freq_i], "Hz"));
		logs_epsrf.push_back(std::make_unique<Common::logCurve>(freq_curve_name, events, num_events, 1, efp_out.oper_freqs[freq_i], "Hz"));
		logs_abs.push_back(std::make_unique<Common::logCurve>(freq_curve_name, events, num_events, 1, efp_out.oper_freqs[freq_i], "Hz"));
		logs_phase.push_back(std::make_unique<Common::logCurve>(freq_curve_name, events, num_events, 1, efp_out.oper_freqs[freq_i], "Hz"));
		logs_re.push_back(std::make_unique<Common::logCurve>(freq_curve_name, events, num_events, 1, efp_out.oper_freqs[freq_i], "Hz"));
	}

	auto db_loc = m_output_db_location;
	db_loc.record_name = m_tool_info.qi_rxo_f_record;
	try
	{
		if (m_cancel_requested)
		{
			return "CANCELLED";
		}

		db_loc.record_name = m_tool_info.qi_rxo_f_record;
		m_curves_repository.SaveCurves(db_loc, logs_rf);
	}
	catch (const std::exception &ex)
	{
		std::string error = std::string(ex.what()) + m_tool_info.qi_rxo_f_record;
		LOG_ERROR(LogLayer::Domain) << error;
		return error;
	}

	LOG_INFO(LogLayer::Domain) << "CurveSet Saved to Insite: " << m_tool_info.qi_rxo_f_record;

	try
	{
		if (m_cancel_requested)
		{
			return "CANCELLED";
		}

		db_loc.record_name = m_tool_info.qi_epsrf_f_record;
		m_curves_repository.SaveCurves(db_loc, logs_epsrf);
	}
	catch (const std::exception &ex)
	{
		std::string error = std::string(ex.what()) + m_tool_info.qi_epsrf_f_record;
		LOG_ERROR(LogLayer::Domain) << error;
		return error;
	}

	LOG_INFO(LogLayer::Domain) << "CurveSet Saved to Insite: " << m_tool_info.qi_epsrf_f_record;

	try
	{
		if (m_cancel_requested)
		{
			return "CANCELLED";
		}

		db_loc.record_name = m_tool_info.qi_z_abs_record;
		m_curves_repository.SaveCurves(db_loc, logs_abs);
	}
	catch (const std::exception &ex)
	{
		std::string error = std::string(ex.what()) + m_tool_info.qi_z_abs_record;
		LOG_ERROR(LogLayer::Domain) << error;
		return error;
	}

	LOG_INFO(LogLayer::Domain) << "CurveSet Saved to Insite: " << m_tool_info.qi_z_abs_record;

	try
	{
		if (m_cancel_requested)
		{
			return "CANCELLED";
		}

		db_loc.record_name = m_tool_info.qi_z_phase_record;
		m_curves_repository.SaveCurves(db_loc, logs_phase);
	}
	catch (const std::exception &ex)
	{
		std::string error = std::string(ex.what()) + m_tool_info.qi_z_phase_record;
		LOG_ERROR(LogLayer::Domain) << error;
		return error;
	}

	LOG_INFO(LogLayer::Domain) << "CurveSet Saved to Insite: " << m_tool_info.qi_z_phase_record;

	try
	{
		if (m_cancel_requested)
		{
			return "CANCELLED";
		}

		db_loc.record_name = m_tool_info.qi_z_re_record;
		m_curves_repository.SaveCurves(db_loc, logs_re);
	}
	catch (const std::exception &ex)
	{
		std::string error = std::string(ex.what()) + m_tool_info.qi_z_re_record;
		LOG_ERROR(LogLayer::Domain) << error;
		return error;
	}

	LOG_INFO(LogLayer::Domain) << "CurveSet Saved to Insite: " << m_tool_info.qi_z_re_record;

	std::vector<std::unique_ptr<Common::logCurve>> logs_st;
	logs_st.reserve(num_pads);

	std::vector<std::unique_ptr<Common::logCurve>> logs_mf;
	logs_mf.reserve(num_pads);

	std::vector<std::unique_ptr<Common::logCurve>> logs_dc;
	logs_dc.reserve(num_pads);

	const auto st_pad_names = m_tool_info.standoff_pad_curve_names;
	for (const auto &pad_name : st_pad_names)
	{
		auto log_st = std::make_unique<Common::logCurve>(pad_name, events, num_events, num_buttons);
		log_st->curveUnits = efp_out.standoff_est->curveUnits;
		logs_st.push_back(std::move(log_st));
	}

	const auto mf_pad_names = m_tool_info.misfit_pad_curve_names;
	for (const auto &pad_name : mf_pad_names)
		logs_mf.push_back(std::make_unique<Common::logCurve>(pad_name, events, num_events, num_buttons));

	const auto dc_pad_names = m_tool_info.compressed_pad_curve_names;
	for (const auto &pad_name : dc_pad_names)
		logs_dc.push_back(std::make_unique<Common::logCurve>(pad_name, events, num_events, num_buttons));

	for (auto pad_i = 0, pad_start = 0; pad_i < num_pads; ++pad_i, pad_start += num_buttons)
	{
		for (auto event_i = 0, event_all_start = 0, event_pad_start = 0;
			 event_i < num_events;
			 ++event_i, event_all_start += num_data_samples, event_pad_start += num_buttons)
		{
			if (m_cancel_requested)
			{
				return "CANCELLED";
			}

			std::copy(efp_out.standoff_est->DataBegin() + event_all_start + pad_start,
				 efp_out.standoff_est->DataBegin() + event_all_start + pad_start + num_buttons,
				 logs_st[pad_i]->DataBegin() + event_pad_start);

			std::copy(efp_out.misfitall->DataBegin() + event_all_start + pad_start,
				 efp_out.misfitall->DataBegin() + event_all_start + pad_start + num_buttons,
				 logs_mf[pad_i]->DataBegin() + event_pad_start);

			std::copy(efp_out.rf_est.back()->DataBegin() + event_all_start + pad_start,
				 efp_out.rf_est.back()->DataBegin() + event_all_start + pad_start + num_buttons,
				 logs_dc[pad_i]->DataBegin() + event_pad_start);
		}
	}
	try
	{
		if (m_cancel_requested)
		{
			return "CANCELLED";
		}

		db_loc.record_name = m_tool_info.qi_standoff_data_record;
		m_curves_repository.SaveCurves(db_loc, logs_st);
	}
	catch (const std::exception &ex)
	{
		std::string error = std::string(ex.what()) + m_tool_info.qi_standoff_data_record;
		LOG_ERROR(LogLayer::Domain) << error;
		return error;
	}

	LOG_INFO(LogLayer::Domain) << "CurveSet Saved to Insite: " << m_tool_info.qi_standoff_data_record;

	try
	{
		if (m_cancel_requested)
		{
			return "CANCELLED";
		}

		db_loc.record_name = m_tool_info.qi_misfit_data_record;
		m_curves_repository.SaveCurves(db_loc, logs_mf);
	}
	catch (const std::exception &ex)
	{
		std::string error = std::string(ex.what()) + m_tool_info.qi_misfit_data_record;
		LOG_ERROR(LogLayer::Domain) << error;
		return error;
	}

	LOG_INFO(LogLayer::Domain) << "CurveSet Saved to Insite: " << m_tool_info.qi_misfit_data_record;

	try
	{
		if (m_cancel_requested)
		{
			return "CANCELLED";
		}

		db_loc.record_name = m_tool_info.qi_dc_rxo_record;
		m_curves_repository.SaveCurves(db_loc, logs_dc);
	}
	catch (const std::exception &ex)
	{
		std::string error = std::string(ex.what()) + m_tool_info.qi_dc_rxo_record;
		LOG_ERROR(LogLayer::Domain) << error;
		return error;
	}

	LOG_INFO(LogLayer::Domain) << "CurveSet Saved to Insite: " << m_tool_info.qi_dc_rxo_record;

	auto rxosaveStatus = SaveFinalRxoEpsrf(std::move(logs_rf), std::move(logs_epsrf), efp_out.oper_freqs);
	if (!rxosaveStatus.empty())
	{
		return rxosaveStatus;
	}
	if (m_cancel_requested)
	{
		return "CANCELLED";
	}

	const auto end_proc = std::chrono::high_resolution_clock::now();
	double time_taken_secs = std::chrono::duration<double>(end_proc - begin_proc).count();

	LOG_INFO(LogLayer::Domain) << "Completed writing curves to Insite and process took " << time_taken_secs;
	return "";
}

std::pair<ComputeStatus, std::string>  QIEFPDomainService::ReadSMCData()
{
	try
	{
		m_smc_data.comp_cal_freq_logs_re.clear();
		m_smc_data.comp_cal_freq_logs_im.clear();
		m_smc_data.cal_freq_logs_re.clear();
		m_smc_data.oper_freqs.clear();
		m_smc_data.mangle_use.clear();
		m_smc_data.epsrm_use.clear();

		auto db_loc_re = m_input_db_location;
		db_loc_re.record_name = m_tool_info.smc_re_record;
		if (db_loc_re.record_name.empty())
		{
			return { ComputeStatus::FAILED, m_tool_info.smc_re_record + " is not available for tool"};
		}

		auto db_loc_im = m_input_db_location;
		db_loc_im.record_name = m_tool_info.smc_im_record;
		if (db_loc_im.record_name.empty())
		{
			return { ComputeStatus::FAILED, m_tool_info.smc_im_record + " is not available for tool"};
		}
		auto db_loc_cre = m_input_db_location;
		db_loc_cre.record_name = m_tool_info.smc_cre_record;
		if (db_loc_cre.record_name.empty())
		{
			return { ComputeStatus::FAILED, m_tool_info.smc_cre_record + " is not available for tool"};
		}

		std::vector<std::string> epsrmCurveNames;
		std::vector<std::string> mangleCurveNames;
		std::vector<std::string> freqCurveNames;
		for (int freq_i = 0; freq_i < m_available_freqs.size(); freq_i++)
		{			

			const auto pad_curve_names = m_tool_info.pad_curve_names[freq_i];
			auto comp_cal_freq_logs_re = m_curves_repository.ReadMultiPadCurve(pad_curve_names, db_loc_re, m_min_depth, m_max_depth);
			if (m_cancel_requested)
			{
				return { ComputeStatus::CANCELLED, "Process Cancelled." };
			}

			m_smc_data.comp_cal_freq_logs_re.push_back(std::move(comp_cal_freq_logs_re));
			auto comp_cal_freq_logs_im = m_curves_repository.ReadMultiPadCurve(pad_curve_names, db_loc_im, m_min_depth, m_max_depth);
			if (m_cancel_requested)
			{
				return { ComputeStatus::CANCELLED, "Process Cancelled." };
			}

			m_smc_data.comp_cal_freq_logs_im.push_back(std::move(comp_cal_freq_logs_im));
			auto cal_freq_logs_re = m_curves_repository.ReadMultiPadCurve(pad_curve_names, db_loc_cre, m_min_depth, m_max_depth);
			if (m_cancel_requested)
			{
				return { ComputeStatus::CANCELLED, "Process Cancelled." };
			}

			m_smc_data.cal_freq_logs_re.push_back(std::move(cal_freq_logs_re));

			epsrmCurveNames.push_back(m_tool_info.smc_epsrm_curve_names[freq_i]);
			

			mangleCurveNames.push_back(m_tool_info.smc_mud_angle_curve_names[freq_i]);

			freqCurveNames.push_back(m_tool_info.frequency_curve_names[freq_i]);
			
			auto freqData = m_curves_repository.ReadCurves(
												   db_loc_re,
												   freqCurveNames,
												   m_min_depth,
												   m_max_depth)
								.front()
								->GetPart(m_min_depth, m_max_depth);
			if (m_cancel_requested)
			{
				return { ComputeStatus::CANCELLED, "Process Cancelled." };
			}

			m_smc_data.oper_freqs.push_back(freqData->data[0]);
			freqCurveNames.clear();
		}

		m_smc_data.epsrm_use = std::move(m_curves_repository.ReadCurves(db_loc_re, epsrmCurveNames, m_min_depth, m_max_depth));
		m_smc_data.mangle_use = std::move(m_curves_repository.ReadCurves(db_loc_re, mangleCurveNames, m_min_depth, m_max_depth));
	}
	catch (const std::exception &ex)
	{
		return { ComputeStatus::FAILED, "Failed to ReadSMCData!" };
	}

	LOG_INFO(LogLayer::Domain) << "Completed Reading SMCData";

	return { ComputeStatus::IN_PROGRESS, "Completed Reading SMCData"};
}

std::pair<ComputeStatus, std::string> QIEFPDomainService::ReadSMCIG()
{
	try
	{
		auto db_loc_ig = m_input_db_location;
		auto used_ig_keys = {m_tool_info.smc_ig_rxo_record, m_tool_info.smc_ig_epsrf_record};
		m_smc_data.init_guess.clear();
		for (const auto &ig_key : used_ig_keys)
		{
			db_loc_ig.record_name = ig_key;
			const auto pad_names = m_tool_info.compressed_pad_curve_names;
			auto init_guess = m_curves_repository.ReadMultiPadCurve(pad_names, db_loc_ig, m_min_depth, m_max_depth);
			if (m_cancel_requested)
			{
				return { ComputeStatus::CANCELLED, "Process Cancelled." };
			}			
			m_smc_data.init_guess[ig_key] = std::move(init_guess);			
		}

		if (m_cancel_requested)
		{
			return { ComputeStatus::CANCELLED, "Process Cancelled." };
		}

		return { ComputeStatus::IN_PROGRESS, "Completed Reading SMCIG"};
	}
	catch (const std::exception &)
	{
		return { ComputeStatus::FAILED, "Failed to ReadSMCIG!" };
	}
}

std::string QIEFPDomainService::SaveFinalRxoEpsrf(std::vector<std::unique_ptr<Common::logCurve>> &&all_rxo_logs,
											 std::vector<std::unique_ptr<Common::logCurve>> &&all_epsrf_logs,
											 const std::vector<double> &freq)
{
	try
	{
		const auto num_pads = m_tool_info.number_of_pads;

		const auto min_freq_it = min_element(freq.begin(), freq.end());
		const auto max_freq_it = max_element(freq.begin(), freq.end());

		const auto final_rxo_start_idx = num_pads * distance(freq.begin(), min_freq_it);
		const auto final_epsrf_start_idx = num_pads * distance(freq.begin(), max_freq_it);

		std::vector<std::unique_ptr<Common::logCurve>> final_rxo_logs;
		std::vector<std::unique_ptr<Common::logCurve>> final_epsrf_logs;

		// main pad data
		auto all_rxo_start_it = all_rxo_logs.begin();
		advance(all_rxo_start_it, final_rxo_start_idx);
		auto all_rxo_end_it = all_rxo_logs.begin();
		advance(all_rxo_end_it, final_rxo_start_idx + num_pads);
		std::move(all_rxo_start_it, all_rxo_end_it, back_inserter(final_rxo_logs));

		auto all_epsrf_start_it = all_epsrf_logs.begin();
		advance(all_epsrf_start_it, final_epsrf_start_idx);
		auto all_epsrf_end_it = all_epsrf_logs.begin();
		advance(all_epsrf_end_it, final_epsrf_start_idx + num_pads);
		std::move(all_epsrf_start_it, all_epsrf_end_it, back_inserter(final_epsrf_logs));

		const auto pad_curve_names = m_tool_info.compressed_pad_curve_names;

		for (auto pad_i = 0; pad_i < num_pads; ++pad_i)
		{
			final_rxo_logs[pad_i]->curveName = pad_curve_names[pad_i];
			final_epsrf_logs[pad_i]->curveName = pad_curve_names[pad_i];
		}

		// frequency of acquisition
		const auto freq_curve_name = m_tool_info.common_frequency_record;
		auto final_rxo_foa_log = std::make_unique<Common::logCurve>(freq_curve_name, final_rxo_logs[0]->events, final_rxo_logs[0]->numberOfEvents, 1, *min_freq_it);

		// averages per pad (EDD)
		const auto edd_names = m_tool_info.edd_curve_names;
		auto final_rxo_edd_logs = CalculateAveragesPerCurve(final_rxo_logs, edd_names);

		// overall average resistivity
		const auto avg_res_name = m_tool_info.avg_resitivity_record;
		auto final_rxo_avg_res_log = CalculateAverage(final_rxo_logs, avg_res_name);

		// std::move all additional curves to final rxo vector
		final_rxo_logs.push_back(std::move(final_rxo_foa_log));
		std::move(final_rxo_edd_logs.begin(), final_rxo_edd_logs.end(), back_inserter(final_rxo_logs));
		final_rxo_logs.push_back(std::move(final_rxo_avg_res_log));

		auto db_loc = m_output_db_location;
		db_loc.record_name = m_tool_info.qi_rxo_data_record;

		m_curves_repository.SaveCurves(db_loc, final_rxo_logs);

		// frequency of acquisition for epsrf
		auto final_epsrf_foa_log = std::make_unique<Common::logCurve>(freq_curve_name, final_epsrf_logs[0]->events, final_epsrf_logs[0]->numberOfEvents, 1, *max_freq_it);
		final_epsrf_logs.push_back(std::move(final_epsrf_foa_log));

		db_loc.record_name = m_tool_info.qi_esprf_data_record;
		m_curves_repository.SaveCurves(db_loc, final_epsrf_logs);
		LOG_INFO(LogLayer::Domain) << "CurveSet Saved to Repository: " << m_tool_info.qi_esprf_data_record;
		return "";
	}
	catch (const std::exception ex)
	{
		return std::string(ex.what()) + " in " + m_tool_info.qi_rxo_data_record;
	}
}

std::vector<std::unique_ptr<Common::logCurve>> QIEFPDomainService::CalculateAveragesPerCurve(const std::vector<std::unique_ptr<Common::logCurve>> &in_curves,
																				   const std::vector<std::string> &avg_names)
{
	if (in_curves.size() != avg_names.size())
		throw std::invalid_argument("in_curves size and avg_names size should be the same");

	std::vector<std::unique_ptr<Common::logCurve>> avg_logs(in_curves.size());
	for (auto i = 0u; i < in_curves.size(); ++i)
	{
		avg_logs[i] = CalculateAverage(*in_curves[i], avg_names[i]);
	}
	return avg_logs;
}

void QIEFPDomainService::SetSMCInitGuess(const Common::logCurve* rxo_ig,
	const Common::logCurve* epsrf_ig,
	BoreholeImagingStudio::QIML::InputDataEFP* inputData)
{
	LOG_INFO(LogLayer::Domain) << "SetSMCInitGuess Start..";
	
	if (rxo_ig == nullptr || epsrf_ig == nullptr)
		throw std::invalid_argument("SMC initial guess is empty");

	auto num_events = m_smc_data.comp_cal_freq_logs_re[0]->numberOfEvents;
	auto total_buttons_number = m_smc_data.comp_cal_freq_logs_re[0]->numberOfDataSamples;	

	if (rxo_ig->numberOfEvents != num_events || epsrf_ig->numberOfEvents != num_events)
		throw std::invalid_argument("SMC initial guesses number of events doesnt matches with other input data events.");

	inputData->xIG.clear();
	inputData->xIG.reserve(num_events * total_buttons_number * 6);

	auto rxo_top_ind = 0;
	auto epsrf_top_ind = 0;
	const auto rxo_data_start = rxo_top_ind * total_buttons_number;
	const auto rxo_data_end = (rxo_top_ind + num_events) * total_buttons_number;

	const auto epsrf_data_start = epsrf_top_ind * total_buttons_number;

	for (auto rxo_data_i = rxo_data_start, epsrf_data_i = epsrf_data_start;
		rxo_data_i < rxo_data_end;
		++rxo_data_i, ++epsrf_data_i)
	{
		inputData->xIG.push_back(0.0);								// Standoff (currently not used)
		inputData->xIG.push_back(rxo_ig->data[rxo_data_i]);		// Rf
		inputData->xIG.push_back(epsrf_ig->data[epsrf_data_i]);	// EpsRf freq 1
		inputData->xIG.push_back(0.0);								// EpsRf freq 2 / EpsRf freq 1 (currently not used)
		inputData->xIG.push_back(0.0);								// EpsRf freq 3 / EpsRf freq 2 (currently not used)
		inputData->xIG.push_back(0.0);								// DC component for Rf (currently not used)
	}

	LOG_INFO(LogLayer::Domain) << "SetSMCInitGuess Completed..";

}
