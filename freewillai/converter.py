import torch
import logging
import tensorflow as tf
from typing import Optional, Tuple

import sys

class ModelConverter:

    @staticmethod
    def keras_to_onnx(
        model, 
        output_path: str, 
        input_size: Optional[Tuple[int, ...]]=None
    ) -> None:
        if input_size is None:
            input_size = [tf.TensorSpec(model.input.shape, tf.float32, name='input')]

        logging.debug(f'Writing the onnx model to {output_path}')
        import tf2onnx
        _ = tf2onnx.convert.from_keras(model, input_size, output_path=output_path)

    @staticmethod
    def sklearn_to_onnx(
        model,
        output_path: str,
        input_size: Optional[Tuple[int, ...]] = None
    ) -> None:
        
        if input_size is None:
            try:
                input_size = model.coef_.shape[1]
            except AttributeError:
                input_size = model.n_features_in_
            except Exception as err:
                raise RuntimeError(err)
            
        logging.debug(f'Writing the onnx model to {output_path}')
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        if isinstance(model, str):
            import pickle
            model =  pickle.load(open(str(model), "rb"))

        initial_type = [('float_input', FloatTensorType([None, input_size]))]
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=17)
        with open(output_path, "wb") as f:
            f.write(onx.SerializeToString())

    @staticmethod
    def pytorch_to_onnx(
        model, 
        output_path: str, 
        input_size: Tuple[int, ...]
    ) -> None:
        logging.debug(f'Writing the onnx model to {output_path}')
        dummy_input = torch.randn(1, *input_size, requires_grad=True)
        model.eval()

        torch.onnx.export(
            model, 
            dummy_input, 
            output_path, 
            input_names = ['input'], 
            output_names = ['output'], 
            dynamic_axes={
                'input' : {0 : 'batch_size'}, 
                'output' : {0 : 'batch_size'}
            },
            verbose=False,
            export_params=True,
        )

    @staticmethod
    def dump_onnx(model, output_path, _) -> None:
        logging.debug(f'Writing the onnx model to {output_path}')
        import onnx
        onnx.checker.check_model(model)
        onnx.save(model, output_path)
