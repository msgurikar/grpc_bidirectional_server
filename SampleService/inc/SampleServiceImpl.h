#pragma once

#include <SampleService.grpc.pb.h>

#include <memory>
#include <atomic>
#include <mutex>


/**
 * SampleServiceImpl class implements SampleService proto contracts.
 */
class SampleServiceImpl : public SampleService::SampleService::Service
{
    public:
    
        SampleServiceImpl();      
        
      
        grpc::Status Compute(
            grpc::ServerContext *context,
            grpc::ServerReaderWriter<SampleService::Response, SampleService::Request> *stream) override;

    private:       

        void ReadRequest(grpc::ServerReaderWriter<SampleService::Response, SampleService::Request>* streamReaderWriter);
        void ProcessRequest(SampleService::Request& request);

    private:                       
        std::atomic<bool> m_process_completed{ false };
        std::mutex m_lock_compute;
};
