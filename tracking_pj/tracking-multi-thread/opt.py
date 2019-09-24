import argparse
parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--camera_parameter_path', default='/file/model/camera_parameter.pickle', help='camera_parameter_path')
parser.add_argument('--webcam', default= '0', help='webcam url list')
parser.add_argument('--inp_dim', dest='inp_dim', type=str, default='608',help='inpdim')
parser.add_argument('--outdir', dest='outputpath',default='/home/fudan/Desktop/zls/tracking_pj/tracking-multi-thread/result/webcam_output', help='output-directory')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)

parser.add_argument('--save_video', dest='save_video',help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis', default=True, help='visualize image')
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--inputResH', default=320, type=int,
                    help='Input image height')
parser.add_argument('--inputResW', default=256, type=int,
                    help='Input image width')
parser.add_argument('--outputResH', default=80, type=int,
                    help='Output heatmap height')
parser.add_argument('--outputResW', default=64, type=int,
                    help='Output heatmap width')
parser.add_argument('--fast_inference', default=True, type=bool,
                    help='Fast inference')
parser.add_argument('--conf', dest='confidence', type=float, default=0.05,
                    help='bounding box confidence threshold')
parser.add_argument('--nms', dest='nms_thesh', type=float, default=0.6,
                    help='bounding box nms threshold')
parser.add_argument('--nClasses', default=33, type=int,
                    help='Number of output channel')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size')
opt = parser.parse_args()

opt.num_classes = 80