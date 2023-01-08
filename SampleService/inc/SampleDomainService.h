#pragma once


#include "CurvesRepository.h"
#include "ToolRepository.h"

#include <OutputDataSMC.h>
#include <IAlgorithmEFP.h>
#include <InputDataEFP.h>
#include <OutputDataEFP.h>
#include <QIEFPToolInfo.h>
#include <logCurve.h>
#include <Logger.h>

#include <vector>
#include <memory>
#include <atomic>


/**
	* EFP Computation Status enum types, to indicate the status of computation to service layer.
	*/
enum ComputeStatus
{
	SUCCESS = 0,
	IN_PROGRESS = 1,
	FAILED = 2,
	CANCELLED = 3
};		

/**
	* QIEFPDomainService class performs QI efp algorithm.
	*/
class QIEFPDomainService
{
	public:
		/**
		* @brief  Constructor
		* @param [in] curves_repository Curve Repository service.
		* @param [in] tool_repository Tool Repository service.
		*/
		QIEFPDomainService(CurvesRepository &&curves_repository, ToolRepository &&tool_repository);

		/**
		* @brief  Initalize the default values.
		* @param [in] db_location DBLocation holds the insite details.
		* @param [in] context Servercontext instance
		* @return Initialization status as ComputeStatus(SUCCESS, FAILED).
		*/
		std::pair<ComputeStatus, std::string> Init(const Common::DBLocation &db_location, std::vector<std::string>& freqs);

		/**
		* @brief Gets tool type for the selected data set.
		* @return Tool type string value for given data set.
		*/
		std::string GetToolType();

		/**
		* @brief  Executes Efp algorithm for given records.
		* @param [in] db_location DBLocation holds the insite details.
		* @param [in] min_depth Minimum depth to execute Efp algorithm.
		* @param [in] max_depth Maximum depth for execute Efp algorithm.
		* @param [in] inputParams input params for execute Efp algorithm.
		* @param [in] streamReaderWriter gRPC Server Reader and writer,sends
			continuous progress to client and reads cancel request from client.
		* @return Algorithm Computation status as gRPC Status.
		*/
		std::pair<ComputeStatus, std::string> Compute(const Common::DBLocation& db_location,
			const double min_depth,
			const double max_depth,
			BoreholeImagingStudio::QIML::EFPParams& inputParams, bool IsComputeChunk,
			std::function<void(ComputeStatus, std::string, float)> progress_cb);

		/**
		* @brief  Executes Efp algorithm for given record chunk.
		* @param [in] db_location DBLocation holds the insite details
		* @param [in] min_depth Minimum depth to execute Efp algorithm.
		* @param [in] max_depth Maximum depth for execute Efp algorithm.
		* @param [in] inputParams input params for execute Efp algorithm.
		* @param [in] streamReaderWriter gRPC Server Reader and writer,sends continuous progress to client
		*	 and reads cancel request from client.
		* @return Algorithm Computation status as gRPC Status.
		*/
		std::pair<ComputeStatus, std::string>  ComputeByChunk(const Common::DBLocation& db_location,
			const double min_depth,
			const double max_depth,
			BoreholeImagingStudio::QIML::EFPParams& inputParams,
			std::function<void(ComputeStatus, std::string, float)> progress_cb);
		

		/**
		* @brief  Sets Cancellation request flag true from service layer.		
		*/
		void SetCancellationRequest();	


	private:		
			

		/**
		* @brief  Prepares input data for Efp Algorithm.
		* @param [in] inputParams input params for Efp algorithm
		* @param [in] streamReaderWriter gRPC Server Reader and
		*	writer,sends continuous progress to client and reads cancel request from client.
		* @return InputDataEfp and error message if there is any.
		*/
		std::pair<std::string, std::unique_ptr<BoreholeImagingStudio::QIML::InputDataEFP>> DataPrepEFP(BoreholeImagingStudio::QIML::EFPParams &inputParams);

		/**
		* @brief  Prepares input data for Efp Algorithm.
		* @param [in] inputParams input params for Efp algorithm
		* @param [in] isCPU Flag to decide whether the algorithm should run in cpu or gpu.
		* @return Efp algorithm result as OutputDataEFP.
		*/
		BoreholeImagingStudio::QIML::OutputDataEFP ComputeEFP(BoreholeImagingStudio::QIML::InputDataEFP &inputDataEFP, bool isCPU);

		/**
		* @brief  Computes Efp Algorithm asynchronously.
		* @param [in] inputDataEFP input data for Efp algorithm.
		* @param [in] isCPU Flag to decide whether the algorithm should run in cpu or gpu.		
		* @return Efp algorithm result as OutputDataEFP.
		*/
		BoreholeImagingStudio::QIML::OutputDataEFP ComputeEFPAsync(BoreholeImagingStudio::QIML::InputDataEFP &inputDataEFP, bool isCPU);

		/**
		* @brief  Saves computed result to insite.
		* @param [in] efp_out Efp algorithm result.		
		* @return Error message if there is any.
		*/
		std::string SaveEFP(BoreholeImagingStudio::QIML::OutputDataEFP &efp_out);

		/**
		* @brief  Reads SMC curves for Data preparation.		
		* @return Curve reading status as Compute status.
		*/
		std::pair<ComputeStatus, std::string> ReadSMCData();

		/**
		* @brief  Reads SMC Initial Guesses curves for Data preparation.		
		* @return Curve reading status as Compute status..
		*/
		std::pair<ComputeStatus, std::string> ReadSMCIG();

		/**
		* @brief Saves the Epsrf curve in insite.
		* @param [in] all_rxo_logs RXO curves.
		* @param [in] all_epsrf_logs Epsrf curves.
		* @param [in] freq Frequencies.
		* @return Error message if there is any.
		*/
		std::string SaveFinalRxoEpsrf(std::vector<std::unique_ptr<Common::logCurve>> &&all_rxo_logs,
										std::vector<std::unique_ptr<Common::logCurve>> &&all_epsrf_logs,
										const std::vector<double> &freq);

		/**
		* @brief Calculates the average per curve.
		* @param [in] in_curves Input curves..
		* @param [in] avg_names Curves names for which average needs to be calculated.
		* @return Log Curve with calculated average values .
		*/
		std::vector<std::unique_ptr<Common::logCurve>> CalculateAveragesPerCurve(
			const std::vector<std::unique_ptr<Common::logCurve>> &in_curves,
			const std::vector<std::string> &avg_names);	

		/**
		* @brief  Sets SMC Initial Guesses data into InputDataEFP
		* @param [in] Rxo ig curve data
		* @param [in] epsrf ig curve data
		* @return void
		*/
		void SetSMCInitGuess(const Common::logCurve* rxo_ig, 
			const Common::logCurve* epsrf_ig,
			BoreholeImagingStudio::QIML::InputDataEFP* inputData);
		

	private:
		CurvesRepository m_curves_repository;
		ToolRepository m_tool_repository;
		BoreholeImagingStudio::QIML::QIEFPToolInfo m_tool_info;
		Common::DBLocation m_input_db_location;
		Common::DBLocation m_output_db_location;
		BoreholeImagingStudio::QIML::OutputDataSMC m_smc_data;
		double m_min_depth;
		double m_max_depth;
		int m_top_ind;
		int m_bot_ind;
		std::vector<std::string> m_available_freqs;
		std::vector<double> m_orig_freqs;
		std::vector<double> m_oper_freqs;		

		double m_pb_compute_percentage = 0.0;
		double m_pb_dataprep_percentage = 0.0;
		double m_pb_writing_percentage = 0.0;

		const std::string ddm_file_name = "ct_model_part3_4_3freq6MAll_linearAbs.mat";
		std::atomic<bool> m_cancel_requested{ false };
		std::atomic<bool> m_process_completed{ false };		
};
