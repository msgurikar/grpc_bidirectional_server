
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <csignal>
#include <iostream>
#include <thread>

#include "SampleServiceImpl.h"


using namespace std;

namespace
{
	const std::string SERVICE_NAME = "SampleService";	
	}


std::recursive_mutex server_mutex;
std::unique_ptr<grpc::Server> server;

//thread function
void doShutdown()
{
	cout << "Entering doShutdown" << endl;

	//getchar();  // press a key to shutdown the thread
	auto deadline = std::chrono::system_clock::now() +
		std::chrono::milliseconds(300);
	server->Shutdown(deadline);
	//server->Shutdown();
	std::cout << "Server is shutting down. " << std::endl;	
}

void signal_handler(int signal_num)
{
	//std::lock_guard<std::recursive_mutex> lock(server_mutex);
	cout << "The interrupt signal is (" << signal_num
		<< "). \n";
	

	switch (signal_num)
	{
	case SIGINT:
		std::puts("It was SIGINT");
	
		break;
	case SIGTERM:
		std::puts("It was SIGTERM");
	
		break;
	default:
		break;
	}

	// It terminates the  program
	cout << "Calling Server Shutdown" << endl;;
	std::thread t = std::thread(doShutdown);
	//server->Shutdown();		
	cout << "Calling exit()" << endl;
	t.join();
	//exit(0);
}






/// <summary>
/// Main method of the console app
/// </summary>
/// <param name="values">holds ip address and port</param>
/// <returns>int</returns>
int main()
{
	std::string address = "0.0.0.0:40056";
	SampleServiceImpl sample_service;

	grpc::EnableDefaultHealthCheckService(true);
	grpc::reflection::InitProtoReflectionServerBuilderPlugin();
	grpc::ServerBuilder builder;

	builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIME_MS, 1000 * 60 * 1);
	builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 1000 * 10);
	builder.AddChannelArgument(GRPC_ARG_HTTP2_MIN_SENT_PING_INTERVAL_WITHOUT_DATA_MS, 1000 * 10);
	builder.AddChannelArgument(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);
	builder.AddChannelArgument(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);

	//TODO: use secure SSL connection
	builder.AddListeningPort(address, grpc::InsecureServerCredentials());
	// Register "service" as the instance through which we'll communicate with
	// clients. In this case it corresponds to an *synchronous* service.
	builder.RegisterService(&sample_service);
	// Finally assemble the server.
	server = builder.BuildAndStart();
	
	/*std::signal(SIGTERM, signal_handler);
	std::signal(SIGSEGV, signal_handler);
	std::signal(SIGINT, signal_handler);
	std::signal(SIGABRT, signal_handler);*/

	// Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
	//std::thread t = std::thread(doShutdown);
	


	// Wait for the server to shutdown. Note that some other thread must be
	// responsible for shutting down the server for this call to ever return.
	cout << "Server waiting " << endl;
	server->Wait();	
	cout << "Server exited " << endl;
	//t.join();
	return 0;
}