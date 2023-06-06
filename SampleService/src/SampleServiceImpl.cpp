#include "SampleServiceImpl.h"

#include <algorithm>
#include <iterator>
#include <chrono>
#include <future>
#include <thread>
#include <string>
#include <vector>
#include <iostream>

#ifdef _WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif

using namespace grpc;
using namespace std;
using namespace SampleService;


SampleServiceImpl::SampleServiceImpl()    
{

}


Status SampleServiceImpl::Compute(
    ServerContext *context,
    ServerReaderWriter<::Response,
                       ::Request> *stream)
{  
    std::lock_guard<std::mutex> lock(m_lock_compute);
    try
    {
        cout << "--------------------------- Start of request-----------------" << endl;
        Response response;
        response.set_status(ResponseStatus::IN_PROGRESS); //Status: -1 in progress
        response.set_message("Request Received.");
        stream->Write(response);
       
        const auto begin_proc = std::chrono::high_resolution_clock::now();
        Request request;
        stream->Read(&request);
        cout << "Request recived " << request.input_msg() << endl;
        m_process_completed = false;
       
        auto readTask = std::async(std::launch::async, &SampleServiceImpl::ReadRequest, this, stream);
       
        std::future<void> result;
        
        std::string message = "In Progress";

        try
        {

            std::future_status compute_status = std::future_status::deferred;
            result = std::async(std::launch::async, &SampleServiceImpl::ProcessRequest, this, std::ref(request));
            do
            {
                compute_status = result.wait_for(std::chrono::seconds(30));
                response.set_status(ResponseStatus::IN_PROGRESS);
                response.set_message(message);
                stream->Write(response);
            } while (compute_status != std::future_status::ready);
        }
        catch (const std::exception& e)
        {
            response.set_status(ResponseStatus::FAILED);
            response.set_message("Failed");
            stream->Write(response);
        }
        catch (...)
        {
            response.set_status(ResponseStatus::FAILED);
            response.set_message("Failed");
            stream->Write(response);
        }
        
        result.wait();
        readTask.wait();
        cout << "Request Completed " << request.input_msg() << endl;
        const auto end_proc = std::chrono::high_resolution_clock::now();
        double time_taken_secs = std::chrono::duration<double>(end_proc - begin_proc).count();        
        cout << "Process completed and took " << time_taken_secs << "seconds" <<endl;
        cout << "--------------------------- End of request-----------------" << endl;
        auto pod_name = std::string(std::getenv("POD_NAME"));
        cout << "Pod name is" << pod_name << endl;
        response.set_pod_name(pod_name);
        response.set_status(ResponseStatus::SUCCESS); 
        response.set_message("Completed.");
        
        stream->WriteLast(response, ::grpc::WriteOptions());
        return Status::OK;
    }
    catch (const std::exception &e)
    {
       cout << "Exception occured in compute " << std::string(e.what());
    }
}

void SampleServiceImpl::ProcessRequest(Request& request)
{   
    cout << "Input message is " << request.input_msg() << endl;
    std::this_thread::sleep_for(std::chrono::minutes(5));
    m_process_completed = true;
    return;
}


void SampleServiceImpl::ReadRequest(ServerReaderWriter<Response, Request>* streamReaderWriter)
{
    Request request;
    try
    {
        while (streamReaderWriter->Read(&request))
        {            
            if (request.request_cancel())
            {              
                cout << "Cancellation requested.";               
                break;
            }

            if (m_process_completed)
            {
                cout << "ReadCancel-- Process completed." << endl;                
                break;
            }
        }
    }
    catch (const std::exception& ex)
    {
       cout << "Exception occured " << std::string(ex.what());
    }
}


