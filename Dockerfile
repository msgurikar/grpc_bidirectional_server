FROM grpcbase AS builder

COPY . /usr/src/samplegrpc
WORKDIR /usr/src/samplegrpc
RUN mkdir build
WORKDIR /usr/src/samplegrpc/build

RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake ..
RUN cmake --build . 

FROM ubuntu:20.04 AS runtime

WORKDIR /usr/local/bin
COPY --from=builder /usr/src/samplegrpc/build/bin ./


EXPOSE 40056
WORKDIR /usr/local/bin
CMD ["./SampleService"]