import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
import cv2
import numpy as np
from PIL import Image


def main():
    grpc_url = "localhost:8001"
    model_name = "deoldify"
    model_version = b'1'
    channel = grpc.insecure_channel(grpc_url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Health
    try:
        request = service_pb2.ServerLiveRequest()
        response = grpc_stub.ServerLive(request)
        print(f"server: {response}")
    except Exception as ex:
        print(ex)

    # Metadata
    request = service_pb2.ServerReadyRequest()
    response = grpc_stub.ServerReady(request)
    print(f"server: {response}")

    request = service_pb2.ModelReadyRequest(name=model_name,
                                            version=model_version)
    response = grpc_stub.ModelReady(request)
    print(f"model: {response}")

    # Configuration
    request = service_pb2.ModelConfigRequest(name=model_name,
                                             version=model_version)
    response = grpc_stub.ModelConfig(request)
    print(f"model config: {response}")

    # Infer
    video = cv2.VideoCapture("grey_video.mp4")
    i = -1
    while video.isOpened():
        i += 1
        # read video
        ret, frame = video.read()

        # construct request
        request = service_pb2.ModelInferRequest()
        request.model_name = model_name
        request.model_version = model_version
        request.id = "some id goes here"

        input = service_pb2.ModelInferRequest().InferInputTensor()
        input.name = "input"
        input.datatype = "FP32"

        # image
        img_np = cv2.imread('sample.jpeg')
        start_h, start_w, _ = img_np.shape
        img_np = cv2.resize(img_np, (256, 256), interpolation=cv2.INTER_CUBIC)
        # scaling
        img_np = np.asarray(img_np, dtype=np.float32)
        img_np = np.transpose(img_np, axes=(2, 0, 1))
        input.shape.extend([1, 3, 256, 256])
        request.inputs.extend([input])

        output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        output.name = "out"
        request.outputs.extend([output])

        request.raw_input_contents.extend([np.expand_dims(img_np, axis=0).tobytes()])

        response = grpc_stub.ModelInfer(request)
        # we already have one image cause of [0] !
        convert = np.frombuffer(response.raw_output_contents[0], dtype=np.float32)
        convert = convert.reshape((3, 256, 256)).copy()
        convert = np.transpose(convert, axes=(1, 2, 0)).astype(np.uint8)
        result_img = Image.fromarray(convert)
        result_img.save("colored.jpeg")


if __name__ == "__main__":
    main()
