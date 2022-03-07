# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import zlib     # [S1-EX1 addition]
import open3d as o3d     # [S1-EX2 addition]

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


num_frame_tmp = 0

# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######     
    #######
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    # ステップ1：キーコールバックでopen3dを初期化し、ウィンドウを作成します
    ##### キーコールバック機能(キーボード受付)持ちのクラスのインスタンス生成
    window_pcl = o3d.visualization.VisualizerWithKeyCallback()
    ##### 上記インスタンスにビジュアライザ(可視化ウィンドウ)を作成
    window_pcl.create_window()
    
    # step 2 : create instance of open3d point-cloud class
    # ステップ2：open3dポイントクラウドクラスのインスタンスを作成します
    pcd = o3d.geometry.PointCloud()

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    # ステップ3：点群を3Dベクトルに変換してpcdインスタンスにポイントを設定します（open3d関数Vector3dVectorを使用）
    pcd.points = o3d.utility.Vector3dVector(pcl[:,0:3])

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    # ステップ4：最初のフレームで、add_geometryを使用してpcdインスタンスを視覚化に追加します。 他のすべてのフレームには、代わりにupdate_geometryを使用してください
    window_pcl.add_geometry(pcd)    
#    global num_frame_tmp
#    if num_frame_tmp == 0:
        ##### add_geometry メソッド：シーンにジオメトリを追加し、対応するシェーダーを作成する関数
#        window_pcl.add_geometry(pcd)
#        num_frame_tmp += 1
#    else:
        ##### update_geometry メソッド：ジオメトリを更新する関数。この関数は、ジオメトリが変更されたときに呼び出す必要があります。それ以外の場合、ビジュアライザーの動作は定義されていません。
#        window_pcl.update_geometry(pcd)
   
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    # ステップ5：点群を視覚化し、右矢印が押されるまでウィンドウを開いたままにします（キーコード262）
    def do_right_arrow(window_pcl):
        window_pcl.close()
#        window_pcl.destroy_window()
    
    ##### register_key_callback メソッド：キー押下イベントのコールバック関数を登録する関数
    ##### 右矢印キー(262)コール時に関数 do_right_arrow を登録
    window_pcl.register_key_callback(262, do_right_arrow)
    
    ##### update_renderer メソッド：レンダリングを通知する機能を更新する
    window_pcl.update_renderer()
    ##### poll_events メソッド：イベントをポーリングする関数
    window_pcl.poll_events()
    ##### run メソッド：ウィンドウをアクティブにする機能。この関数は、ウィンドウが閉じられるまで現在のスレッドをブロックします。
#    window_pcl.run()    


    #######
    ####### ID_S1_EX2 END #######     
       

# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######     
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    # ステップ1：屋根に取り付けられたLIDARのLIDARデータと範囲画像を抽出します
    ###[C1-3-2, C1-5-1] frame から lidar_name が一致する LiDARオブジェクトを抽出してリストにする
    lidars = [obj for obj in frame.lasers if obj.name == lidar_name]
    lidar = lidars[0]
    
    # step 2 : extract the range and the intensity channel from the range image
    # ステップ2：画像範囲から範囲と強度チャネルを抽出します
    ###[C1-5-1]画像範囲(RangeImage)取得
    ri = []
    ##### データ探索
    if len(lidar.ri_return1.range_image_compressed) > 0: 
        ##### 解凍。Waymoオープンデータセットの説明に記載の方法で解凍する
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    
    # step 3 : set values <0 to zero
    # ステップ3：値<0をゼロに設定
    ###[C1-5-4]
    ri[ri<0] = 0.0
    
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    # ステップ4：範囲チャネルを8ビットスケールにマッピングし、値の全範囲が適切に考慮されていることを確認します
    ###[C1-5-4]8ビットスケール変換
    ##### 画像範囲から範囲(range)を取得
    ri_range = ri[:,:,0]
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
    img_range = ri_range.astype(np.uint8)
    
    ##### 360度の画像範囲における45度の大きさ deg45、中心(180度)位置 ri_center を算出
#    deg45 = int(img_range.shape[1] / 8)
#    ri_center = int(img_range.shape[1]/2)
    ##### 画像範囲を中心から±45度とする
#    img_range = img_range[:,ri_center-deg45:ri_center+deg45]
        
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    # ステップ5：強度チャネルを8ビットスケールにマッピングし、1パーセンタイルと99パーセンタイルの差で正規化して、外れ値の影響を軽減します。
    ##### 画像範囲から光強度(intensity)を取得
    ri_intensity = ri[:,:,1]
    ##### 1パーセンタイルと99パーセンタイルの値取得
    ri_intensity_prcnt01 = np.percentile(ri_intensity, 1)
    ri_intensity_prcnt99 = np.percentile(ri_intensity, 99)
    
    ##### 最大値、最小値をパーセンタイルの値に変更
    ri_intensity[ri_intensity > ri_intensity_prcnt99] = ri_intensity_prcnt99
    ri_intensity[ri_intensity < ri_intensity_prcnt01] = ri_intensity_prcnt01

#    ri_intensity = np.amax(ri_intensity)/2 * ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity)) 
    ri_intensity = ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity)) 
    img_intensity = ri_intensity.astype(np.uint8)

    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    # ステップ6：np.vstackを使用して範囲と強度の画像を垂直方向にスタックし、結果を符号なし8ビット整数に変換します
    ##### 画像範囲と強度画像を垂直方向に結合する
    img_range_intensity = np.vstack((img_range, img_intensity))
    img_range_intensity = img_range_intensity.astype(np.uint8)
    
#debug
#    cv2.imshow('img_intensity', img_intensity)
#    cv2.imshow('range_image', img_range)
#    cv2.imshow('img_range_intensity', img_range_intensity)
#    cv2.waitKey(0)
    
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    # 検出領域の外側で反射率が低すぎるLIDARポイントを削除します
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    # 隣接するピクセルの0から255への反転を回避するために、グランドプレーンのレベルをシフトします
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    # センサー座標をbev-map座標に変換します（中央は下中央）
    
    ####### ID_S2_EX1 START #######     
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    ## ステップ1：x-rangeをbev-imageの高さで割ってbev-mapの離散化を計算します（構成を参照）
    ###[C2-3-2]
    ##### bevのオーダーを取得。検出領域の内側(境界の上限-下限)をbevの高さで除算する
    bev_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
  
    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates    
    ## ステップ2：LIDAR pclのコピーを作成し、すべてのmetrix座標をbev-image座標に変換します
    lidar_pcl_cpy = np.copy(lidar_pcl)
    ##### lidar_pcl[:, 0] ：画像範囲のデータが格納。x座標。
    ##### 画像範囲をbevのオーダーで割り、切り上げ、整数化
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret))

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    # ステップ3：y座標に対してステップ2と同じ操作を実行しますが、負のbev座標が発生しないことを確認します
    # transform all metrix y-coordinates as well but center the foward-facing x-axis on the middle of the image
    ##### lidar_pcl[:, 1] ：画像範囲のデータが格納。y座標。負の座標を解消するバイアス追加。
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret) + (configs.bev_width + 1) / 2)

    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    # ステップ4：前のタスクの関数show_pclを使用して点群を視覚化する
    show_pcl(lidar_pcl_cpy)
    
    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######     
    #######
    print("student task ID_S2_EX2")
    
    ###[C2-3-2]
    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    ## ステップ1：BEVマップと同じ次元のゼロで満たされたnumpy配列を作成します
#    intensity_map = np.zeros(lidar_pcl_cpy.shape)
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    # ステップ2：最初にx、次にy、次に-zで並べ替えて、lidar_pcl_cpyの要素を再配置します（numpy.lexsortを使用）
    lidar_pcl_cpy[lidar_pcl_cpy[:,3]>1.0,3] = 1.0
    idx_intensity = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_cpy = lidar_pcl_cpy[idx_intensity]
#    lidar_pcl_cpy = lidar_pcl_cpy[i, :] for i in sorted_ind
    
    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    ## ステップ3：最上位のz座標のみが保持されるように、xとyが同一のすべてのポイントを抽出します（numpy.uniqueを使用）
    ## また、次のタスクで使用するために、x、yセルあたりのポイント数を「counts」という名前の変数に格納します
    _, indices, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_pcl_int = lidar_pcl_cpy[indices]
    
    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    ## ステップ4：lidar_top_pclの各一意のエントリの強度値を強度マップに割り当てます
    ## 対象のオブジェクト（車両など）がはっきりと見えるように強度がスケーリングされていることを確認してください
    ## また、ポイントクラウド内の最小値と最大値との差の強度を正規化することにより、外れ値の影響が軽減されることを確認してください。    
    intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = lidar_pcl_int[:, 3] / (np.amax(lidar_pcl_int[:, 3])-np.amin(lidar_pcl_int[:, 3]))
    img_intensity = intensity_map * 256
    img_intensity = img_intensity.astype(np.uint8)
        
    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    ## ステップ5：OpenCVを使用して強度マップを一時的に視覚化し、車両が背景から十分に分離していることを確認します
#    cv2.imshow('img_intensity', img_intensity)
#    cv2.waitKey(0)
    
    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    # BEVマップの高さレイヤーを計算する
    ####### ID_S2_EX3 START #######     
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    ## ステップ1：BEVマップと同じ次元のゼロで満たされたnumpy配列を作成します
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))   
    
    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    ## ステップ2：lidar_top_pclの各一意のエントリの高さの値を高さマップに割り当てます
    ## 各エントリが構成ファイルで定義された高さの上限と下限の差で正規化されていることを確認してください
    ## 前のタスクのlidar_pcl_topデータ構造を使用して、height_mapのピクセルにアクセスします
    ##### ソートと同一ポイント削除は、強度で行った結果を流用するため不要
    lidar_pcl_top = lidar_pcl_int
    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))

    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    ## ステップ3：OpenCVを使用して強度マップを一時的に視覚化し、車両が背景から十分に分離していることを確認します
    img_height = height_map * 256
    img_height = img_height.astype(np.uint8)
#    cv2.imshow('img_height', img_height)
#    cv2.waitKey(0)
    

    #######
    ####### ID_S2_EX3 END #######       

    # TODO remove after implementing all of the above steps
#    lidar_pcl_cpy = []
#    lidar_pcl_top = []
#    height_map = []
#    intensity_map = []

    # Compute density layer of the BEV map
    # BEVマップの密度レイヤーを計算する
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    # 個々のマップから3チャネルのbev-mapをアセンブルします
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    # テンソルに変換する前にbev_mapの次元を展開します
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps
