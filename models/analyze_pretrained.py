
from models.predict_by_patches import predict_by_patches
from benchmark.analyzeAll import AnalyzeArchitecture
import cv2

def test():
    #rename output by hand!
    #data_dir = 'E:/pavement inspection/datasets/crack500_out_0.25percent_size/'
    #data_dir = 'E:/pavement inspection/datasets/CrackForestdatasets_output/'
    data_dir = 'E:/pavement inspection/datasets/crack500_out_0.25percent_size/'

    weights = []
    weights_path = 'E:/pavement inspection/notPretrained/crack500/'
    weights.append(weights_path)
    for weight in weights:
        print(weight)
        predict_by_patches(weight, data_dir + 'Test/', True)
        cv2.destroyAllWindows()
        print('Analyze ' + weight)
        #statistics
        AnalyzeArchitecture(weight + 'output/', data_dir + 'Test/')
        cv2.destroyAllWindows()
        print('End of ' + weight)

def main():
    test()

if __name__ == "__main__":
    main()
