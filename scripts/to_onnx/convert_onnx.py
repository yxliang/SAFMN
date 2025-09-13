import os
import cv2 
import glob
import numpy as np 
import onnx
import onnxruntime as ort
import torch  
import torch.onnx 
from basicsr.archs.safmn_arch import SAFMN
import onnxslim

def convert_onnx(model, output_folder, is_dynamic_batches=False): 
    model.eval() 

    fake_x = torch.rand(1, 3, 560, 768, requires_grad=False)
    output_name = os.path.join(output_folder, 'SAFMN_560_768_x5.onnx')
    dynamic_params = None
    if is_dynamic_batches:
        # dynamic_params = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        dynamic_params = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }

    # Export the model   
    torch.onnx.export(model,            # model being run 
        fake_x,                         # model input (or a tuple for multiple inputs) 
        output_name,                    # where to save the model  
        export_params=True,             # store the trained parameter weights inside the model file 
        opset_version=17,               # the ONNX version to export the model to 
        do_constant_folding=True,       # whether to execute constant folding for optimization 
        input_names = ['input'],        # the model's input names 
        output_names = ['output'],      # the model's output names 
        dynamic_axes=dynamic_params) 
   
    try:
        onnxslim.slim(output_name, output_name, model_check=True, constant_folding=True, opt_level=3)
        print(f"ONNX model has been slimmed and saved to {output_name}")
    except Exception as e:
        print(f"An error occurred during ONNX slimming: {e}")

    # fake_x = torch.rand(1, 3, 256, 256, requires_grad=False)
    # output_name = os.path.join(output_folder, 'SAFMN_256_256_x5.onnx')
    # # Export the model   
    # torch.onnx.export(model,            # model being run 
    #     fake_x,                         # model input (or a tuple for multiple inputs) 
    #     output_name,                    # where to save the model  
    #     export_params=True,             # store the trained parameter weights inside the model file 
    #     opset_version=17,               # the ONNX version to export the model to 
    #     do_constant_folding=True,       # whether to execute constant folding for optimization 
    #     input_names = ['input'],        # the model's input names 
    #     output_names = ['output'],      # the model's output names 
    #     dynamic_axes=dynamic_params) 
    # try:
    #     onnxslim.slim(output_name, output_name, model_check=True, constant_folding=True, opt_level=3)
    #     print(f"ONNX model has been slimmed and saved to {output_name}")
    # except Exception as e:
    #     print(f"An error occurred during ONNX slimming: {e}")

    # fake_x = torch.rand(1, 3, 2000, 3000, requires_grad=False) #2800, 3840
    # output_name = os.path.join(output_folder, 'SAFMN_2800_3840_x5.onnx')
    # # Export the model   
    # torch.onnx.export(model,            # model being run 
    #     fake_x,                         # model input (or a tuple for multiple inputs) 
    #     output_name,                    # where to save the model  
    #     export_params=True,             # store the trained parameter weights inside the model file 
    #     opset_version=17,               # the ONNX version to export the model to 
    #     do_constant_folding=True,       # whether to execute constant folding for optimization 
    #     input_names = ['input'],        # the model's input names 
    #     output_names = ['output'],      # the model's output names 
    #     dynamic_axes=dynamic_params) 
    # try:
    #     onnxslim.slim(output_name, output_name, model_check=True, constant_folding=True, opt_level=3)
    #     print(f"ONNX model has been slimmed and saved to {output_name}")
    # except Exception as e:
    #     print(f"An error occurred during ONNX slimming: {e}")

    print('Model has been converted to ONNX')


def convert_pt(model, output_folder): 
    model.eval() 

    fake_x = torch.rand(1, 3, 560, 768, requires_grad=False)
    output_name = os.path.join(output_folder, 'SAFMN_560_768_x5.pt')

    traced_module = torch.jit.trace(model, fake_x)
    traced_module.save(output_name)
    print('Model has been converted to pt')


def test_onnx(onnx_model, input_path, save_path):
    # for GPU inference
    ort_session = ort.InferenceSession(onnx_model, providers=['CUDAExecutionProvider'])

    # ort_session = ort.InferenceSession(onnx_model)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(input_path, '*')))):
        # read image
        imgname = os.path.splitext(os.path.basename(path))[0]

        print(f'Testing......idx: {idx}, img: {imgname}')

        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        [width, height] = img.shape[1], img.shape[0]
    
        if img.size != (768, 560):
            img = cv2.resize(img, (768, 560), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB, HWC -> CHW
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)

        output = ort_session.run(None, {"input": img})

        # save image
        print('Saving!')
        output = np.squeeze(output[0], axis=0)
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            
        output = (output.clip(0, 1) * 255.0).round().astype(np.uint8)
        if(output.shape[1] != width or output.shape[0] != height):
            output = cv2.resize(output, (width*5, height*5), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(save_path, f'{imgname}_results.jpg'), output)


if __name__ == "__main__":
    # model = model = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=2) 

    # pretrained_model = 'experiments/pretrained_models/SAFMN_L_Real_LSDIR_x2.pth'
    model = model = SAFMN(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=5) 
    pretrained_model = 'experiments/pretrained_models/net_g_5500.pth'
    model.load_state_dict(torch.load(pretrained_model)['params'], strict=True)

    ###################Onnx export#################
    output_folder = 'scripts/convert' 

    convert_onnx(model, output_folder, True)
    # convert_pt(model, output_folder)

    ###################Test the converted model #################
    onnx_model = 'scripts/convert/SAFMN_560_768_x5.onnx'
    input_path = '../Datasets/SuperResolution/DeepSR_real/DF2K_val_LR_real/X5'#'datasets/real_test'
    save_path = 'results/onnx_results'
    test_onnx(onnx_model, input_path, save_path)
