import cv2
import numpy as np

import triton_python_backend_utils as pb_utils
import json


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = json.loads(args['model_config'])
        self.output_config = pb_utils.get_output_config_by_name(self.model_config,
                                                                "deoldify_postprocessing_output")
        self.output_type = pb_utils.triton_string_to_numpy(
            self.output_config['data_type']
        )

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []
        # Every Python backend must iterate over every one of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            inp = pb_utils.get_input_tensor_by_name(request, "deoldify_postprocessing_input")
            img = inp.as_numpy()
            W, H = 960, 540

            img = np.transpose(img, axes=(1, 2, 0)).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

            out_tensor = pb_utils.Tensor("deoldify_postprocessing_output", img.astype(self.output_type))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            #
            # Because the infer_response of the models contains the final
            # outputs with correct output names, we can just pass the list
            # of outputs to the InferenceResponse object.
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
