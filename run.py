import gdown
import warnings
warnings.filterwarnings(action = 'ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import hyperspy.api as hs
import numpy as np
from tqdm import tqdm
from core.attention_unet import AttentionUNet
from sklearn.cluster import DBSCAN

def set_parameter():
    while True:
        print("Type the data directory")
        data_dir = input(" >>> ")
        try:
            os.listdir(data_dir)
            break
        except:
            print(f"Check the data directory {data_dir}")
            continue

    while True:
        print("Type the result data directory")
        result_dir = input(" >>> ")
        try:
            os.mkdir(result_dir)
            break
        except:
            break

    while True:
        print("Calculate particle distribution? (y/n)")
        cal_dist = input(" >>> ")
        if (cal_dist == "y")|(cal_dist == "Y"):
            cal_dist = True
            break
        elif (cal_dist == "n")|(cal_dist == "N"):
            cal_dist = False
            break
        else:
            print("Type y or n")
            continue

    while True:
        print("Use cuda? (y/n)")
        device = input(" >>> ")
        if (device == "y")|(device == "Y"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            break
        elif (device == "n")|(device == "N"):
            device = torch.device("cpu")
            break
        else:
            print("Type y or n")
            continue

    return data_dir, result_dir, cal_dist, device


def load_model():
    model_dir = "model/Attention_UNet.pt"
    net = AttentionUNet(img_ch = 1, output_ch = 3)
    try:
        net.load_state_dict(torch.load(model_dir))
    except:
        gdown.download("https://drive.google.com/uc?id=1OyxqKQVX-k3_2ijbQBhQLGbcgsDPb6l4", 
                        "model/Attention_UNet.pt")
        net.load_state_dict(torch.load(model_dir))
    
    return net

def run_model(input_data_dir, model, device, cal_dist):
    with torch.no_grad():
        model.eval()
        input_sig = hs.load(input_data_dir)
        scale = input_sig.axes_manager[0].scale
        input_data = (input_sig - input_sig.data.mean()) / input_sig.data.std()
        input_data = torch.from_numpy(input_data.data.copy())
        input_data = input_data.view(-1, 1, input_data.shape[0], input_data.shape[1])
        input_data = input_data.to(device).float()
        output = model(input_data)
        output = torch.argmax(output, dim = 1)
        output = output.cpu().detach().numpy()[0]
        output = hs.signals.Signal2D(output)
        cluster_input = np.array(np.where(output.data > 0)).T
        cluster_model = DBSCAN(eps = 2, min_samples = 12)
        cluster_model.fit(cluster_input)
        instance_seg_map = np.zeros((1024, 1024))
        instance_seg_map[np.where(output.data > 0)] = cluster_model.labels_ + 1
        instance_seg_map = hs.signals.Signal2D(instance_seg_map)
        label_max = int(instance_seg_map.data.max() + 1)

        seg_map = np.zeros((1024, 1024))

        for i in range(1, label_max):
            label1 = (output.data[instance_seg_map.data == i] == 1).sum()
            label2 = (output.data[instance_seg_map.data == i] == 2).sum()
            if label1 > label2:
                seg_map[instance_seg_map.data == i] = 1
            else:
                seg_map[instance_seg_map.data == i] = 2

        seg_map = hs.signals.Signal2D(seg_map)
        seg_map.change_dtype("uint8")

        single_area = []
        agg_area = []
        if cal_dist:
            single_distance = []
            agg_distance = []

        for i in range(1, label_max):
            area = (instance_seg_map.data == i)
            if np.unique(seg_map.data[np.where(area)]).item() == 1:
                single_area.append(area.sum() * scale * scale)
                current_label = 1
            elif np.unique(seg_map.data[np.where(area)]).item() == 2:
                agg_area.append(area.sum() * scale * scale)
                current_label = 2
            if cal_dist:
                cluster0 = np.array(np.where(area)).T
                for dist_i in range(i + 1, label_max):
                    cluster1 = np.where(instance_seg_map.data == dist_i)
                    if np.unique(seg_map.data[cluster1]).item() == current_label:
                        distance_min = 99999999999
                        cluster1 = np.array(cluster1).T
                        for pixel_i in range(len(cluster1)):
                            min_dist = ((cluster0 - cluster1[pixel_i])**2).sum(axis = 1).min()
                            if distance_min > min_dist:
                                distance_min = min_dist
                        if current_label == 1:
                            single_distance.append(np.sqrt(distance_min).item() * scale)
                        elif current_label == 2:
                            agg_distance.append(np.sqrt(distance_min).item() * scale)


        single_area = np.array(single_area)
        agg_area = np.array(agg_area)
        single_distance = np.array(single_distance)
        agg_distance = np.array(agg_distance)
        
        return seg_map, single_area, agg_area, single_distance, agg_distance



def main():
    dataset_dir, result_dir, cal_dist, device = set_parameter()
    model = load_model()
    model.to(device)
    dataset_list = os.listdir(dataset_dir)
    for data_dir in tqdm(dataset_list):
        if (data_dir[-4:] == ".dm4")|(data_dir[-4:] == ".dm3")|(data_dir[-4:] == ".tif"):
            input_data_dir = f"{dataset_dir}/{data_dir}"
            seg_map, single_area, agg_area, single_distance, agg_distance = run_model(input_data_dir = input_data_dir,
                                                                                      model = model,
                                                                                      device = device,
                                                                                      cal_dist = cal_dist)
            seg_map.save(f"{result_dir}/{data_dir[:-4]}.tif")
            np.savetxt(f"{result_dir}/{data_dir[:-4]}_single_area.csv", single_area, delimiter = ',')
            np.savetxt(f"{result_dir}/{data_dir[:-4]}_agg_area.csv", agg_area, delimiter = ',')
            if cal_dist:
                np.savetxt(f"{result_dir}/{data_dir[:-4]}_single_dist.csv", single_distance, delimiter = ',')
                np.savetxt(f"{result_dir}/{data_dir[:-4]}_agg_dist.csv", agg_distance, delimiter = ',')
                
                
if __name__ == "__main__":
    main()
