import numpy as np
import matplotlib.path as mplPath
import cv2
import os
import json
import sys

PATH = "/content/HCMCAIC"
PATH_ROI = os.path.join(PATH, 'data/roi')
PATH_MOI = os.path.join(PATH, 'data/moi')
PATH_TRACKING = os.path.join(PATH, 'info_tracking')
PATH_VIDEO = os.path.join(PATH, 'data/videos')
PATH_RESULT = os.path.join(PATH, 'results')
PATH_VISUALIZE = os.path.join(PATH,'counting_visualize')
PATH_FRAME_OFFSET = os.path.join(PATH, 'data/frame_offset.json')

def load_roi():
    roi_list = {}
    print("Extracting ROI from {}".format(PATH_ROI))
    for file_name in os.listdir(PATH_ROI):
        with open(os.path.join(PATH_ROI, file_name)) as jsonfile:
            dd = json.load(jsonfile)
            polygon = [(int(x), int(y)) for x, y in dd['shapes'][0]['points']]
            roi_list[file_name[:-5]] = polygon
    return roi_list

def load_moi():
    moi_list = {}
    print("Extracting MOI from {}".format(PATH_MOI))
    for folder_name in os.listdir(PATH_MOI):
        moi_list[folder_name] = {}
        for file_name in os.listdir(os.path.join(PATH_MOI, folder_name)):
            if file_name.endswith(".npy"):
                movement_id = file_name.split("_")[-1][:-4]
                full_path_name = os.path.join(PATH_MOI, folder_name, file_name)
                content = np.load(full_path_name)
                moi_list[folder_name][movement_id] = content
    return moi_list


def load_frame_offset():
    frame_offset_list = {}
    print("Extracting frame offset from {}".format(PATH_FRAME_OFFSET))
    with open(PATH_FRAME_OFFSET) as jsonfile:
        info = json.load(jsonfile)
        for file_name in info:
            offset = {}
            for item in info[file_name]:
                direction = int(item["label"][-1])
                val = int(item["offset"])
                offset[direction] = val
            frame_offset_list[file_name] = offset
    return frame_offset_list


def out_of_roi(center, poly):
    path_array = []
    for poly_point in poly:
        path_array.append([poly_point[0], poly_point[1]])
    path_array = np.asarray(path_array)
    polyPath = mplPath.Path(path_array)

    return polyPath.contains_point(center, radius = 0.5)

def validate_center(center, use_off_set, roi_list):
    const = 5
    off_set = [(const, const), (const, -const), (-const, const), (-const, -const)]

    if not use_off_set:
        return  out_of_roi(center, roi_list)

    for each_off_set in off_set:
        center_change = (center[0] + each_off_set[0], center[1] + each_off_set[1])
        if out_of_roi(center_change, roi_list):
            return False
    return True


def out_of_range_bbox(tracking_info, width, height, off_set):
    x_min = int(tracking_info[4])
    y_min = int(tracking_info[5])
    x_max = int(tracking_info[6])
    y_max = int(tracking_info[7])
    return ((x_min-off_set <=0) or (y_min-off_set<=0) or (x_max+off_set>=width) or (y_max+off_set>=height))

def find_latest_object_and_vote_direction(frame_id_list, cur_fr_id, tracking_info, delta_fix, target_obj_id, roi_list, width, height, use_off_set):
    exist_latest_obj = False
    count_out = 0
    count_in = 0
    offset = 3
    for delta in range(1, delta_fix):
        pre_index = np.where(frame_id_list == (cur_fr_id - delta))[0]
        for each_pre_index in pre_index:
            if tracking_info[each_pre_index][3]==target_obj_id:
                exist_latest_obj = True
                pre_obj_center = center_box(tracking_info[each_pre_index][4:])
                if out_of_range_bbox(tracking_info[each_pre_index], width, height, offset):
                    # print(each_pre_index)
                    count_out += 1
                else:
                    if validate_center(pre_obj_center, False, roi_list):
                        # print(each_pre_index)
                        count_in += 1
                    else:
                        count_out += 1
    return count_out, count_in, exist_latest_obj

def center_box(cur_box):
    return (int((cur_box[0]+cur_box[2])/2), int((cur_box[1]+cur_box[3])/2))


def voting(point, vote_movement, obj_id, moi_list):
    # one vote for each time point lying inside MOI
    exist_MOI = False
    if obj_id not in vote_movement:
        vote_movement[obj_id] = {}
    for moi_id in moi_list:
        moi_content = moi_list[moi_id]
        r, g, b, _ = moi_content[int(point[1])][int(point[0])]
        if r != 0 or g != 0 or b != 0:
            if moi_id not in vote_movement[obj_id]:
                vote_movement[obj_id][moi_id] = 0
            vote_movement[obj_id][moi_id] += 1
            exist_MOI = True
        #    print(obj_id, moi_id)
    return vote_movement, exist_MOI


def build_text_name_dict(moi_list):
    results = {}
    for moi_id in moi_list:
        text_name = moi_id+"_"+"1@"
        results[text_name] = 0
        text_name = moi_id+"_"+"2@"
        results[text_name] = 0
        text_name = moi_id+"_"+"3@"
        results[text_name] = 0
        text_name = moi_id+"_"+"4@"
        results[text_name] = 0
    results = dict(sorted (results.items()))
    return results

def draw_text_summarize(annotate_fr, text_name_dict, width, height):
    x_coor = 20
    y_coor = 20
    for text_name in text_name_dict:
        str_write = text_name+":"+str(text_name_dict[text_name])+"."
        cv2.putText(annotate_fr, str_write, (x_coor, y_coor), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA)
        x_coor += 150
        if x_coor + 150 >=width:
            x_coor = 20
            y_coor += 40

    return annotate_fr

def draw_roi(roi_list, image):
    start_point = roi_list[0]
    for end_point in roi_list[1:]:
        cv2.line(image, start_point, end_point, (0,0,255), 1)
        start_point = end_point

    return image

def draw_moi(annotated_frame, vid_name):
    vid_name_json = vid_name.split("_")
    vid_name_json = vid_name_json[0]+"_"+vid_name_json[1]+".json"
    json_file = open(os.path.join('./moi/cam_01/', vid_name_json))
    content = json.load(json_file)[u'shapes']
    for element in content:
        list_points = element['points']
        label = element['label']
        # draw arrow
        for index, point in enumerate(list_points):
            start_point = (int(list_points[index][0]), int(list_points[index][1]))
            end_point = (int(list_points[index+1][0]), int(list_points[index+1][1]))
            if index+2 == len(list_points):
                cv2.arrowedLine(annotated_frame, start_point, end_point, (255, 0, 0), 2)
                break
            else:
                cv2.line(annotated_frame, start_point, end_point, (255,0,0), 2)
        cv2.putText(annotated_frame, label, end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA)
    return annotated_frame

def car_counting(cam_name, roi_list, moi_list, use_offset_list):
    video_name = cam_name+".mp4"
    print("Processing", cam_name)
    tracking_info = np.load(os.path.join(PATH_TRACKING, 'info_' + video_name + '.npy'), allow_pickle = True)
    N = tracking_info.shape[0]
    frame_id = tracking_info[:, 1].astype(np.int).reshape(N)
    obj_id = tracking_info[:, 3].astype(np.int).reshape(N)
    results = []
    num_car_out = 0
    VISUALIZED = True
    if VISUALIZED:
        input = cv2.VideoCapture(os.path.join(PATH_VIDEO, cam_name + '.mp4'))
        height = int(input.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(input.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        width = 1280
        height = 720
    print("Visualizing ", cam_name)
    already_count = []
    vote_movement = {} # each key is obj_id, each value is A dictionary. Each key in A is movement_id, each value in A is count vote for that object to the corresponding 
    num_1_out = {} # each key is mv_id, each value count number of car 
    num_2_out = {} # each key is mv_id, each value count number of truck
    num_3_out = {} # each key is mv_id, each value count number of car 
    num_4_out = {} # each key is mv_id, each value count number of truck

    use_off_set = True if cam_name in use_offset_list else False
    if use_off_set == True:
        print("Using DSO with offset")
    else:
        print("using DSO")
    total_object = set()
    tracking = {}
    counts = {}
    tracking_mov = {'1': 0, '2': 0, '3':0, '4':0, '5':0, '6':0,'7':0, '8':0, '9':0, '10':0, '11':0,'12':0}
    delta_fix = 10
    num_frame_ths = 0
    for fr_id in range(1, max(frame_id)+1):
        index_cur_fr = np.where(frame_id==fr_id)[0]
        for index_box in index_cur_fr:
            cur_box = tracking_info[index_box][4:]
            cur_center = center_box(cur_box)
            cur_obj_id = tracking_info[index_box][3]
            total_object.add(cur_obj_id)
            if cur_obj_id not in counts:
                counts[cur_obj_id] = 0
            counts[cur_obj_id] += 1
            #print(cur_obj_id)

            is_inside_roi = validate_center(cur_center, use_off_set, roi_list)
            vote_movement, inside_MOI = voting(cur_center, vote_movement, cur_obj_id, moi_list)
            if not inside_MOI:
                continue 
            if not is_inside_roi:# current car is outside roi
                count_out, count_in, is_ok = find_latest_object_and_vote_direction(frame_id, fr_id, tracking_info, 10, cur_obj_id, roi_list, width, height, use_off_set)
                if is_ok: #exist object
                    # pre_obj_center = center_box(latest_obj[4:])
                    # is_inside_roi_pre_obj = validate_center(pre_obj_center, False, roi_list)
                    # print(count_in, count_out)
                    if  counts[cur_obj_id] > num_frame_ths and count_in >= count_out and cur_obj_id not in already_count: # previous car lies inside roi
                        already_count.append(cur_obj_id)
                        max_movement_id = max(vote_movement[cur_obj_id], key=vote_movement[cur_obj_id].get)
                        #print(vote_movement)
                        tracking_mov[str(max_movement_id)] += 1
                        if tracking_info[index_box][0] == 1:
                            if max_movement_id not in num_1_out:
                                num_1_out[max_movement_id] = 0
                            num_1_out[max_movement_id] += 1
                            num_object_out = num_1_out[max_movement_id]
                            class_type = "Type 1"
                        elif tracking_info[index_box][0] == 2:
                            if max_movement_id not in num_2_out:
                                num_2_out[max_movement_id] = 0
                            num_2_out[max_movement_id] += 1
                            num_object_out = num_2_out[max_movement_id]
                            class_type = "Type 2"
                        elif tracking_info[index_box][0] == 3:
                            if max_movement_id not in num_3_out:
                                num_3_out[max_movement_id] = 0
                            num_3_out[max_movement_id] += 1
                            num_object_out = num_3_out[max_movement_id]
                            class_type = "Type 3"
                        elif tracking_info[index_box][0] == 4:
                            if max_movement_id not in num_4_out:
                                num_4_out[max_movement_id] = 0
                            num_4_out[max_movement_id] += 1
                            num_object_out = num_4_out[max_movement_id]
                            class_type = "Type 4"
                        results.append([fr_id, num_object_out, cur_center[0], cur_center[1], max_movement_id, class_type])
            else : # using offset to refine again
                is_out = out_of_range_bbox(tracking_info[index_box], width, height, 2)
                if is_out:
                    count_out, count_in, is_ok = find_latest_object_and_vote_direction(frame_id, fr_id, tracking_info, 10, cur_obj_id, roi_list, width, height, use_off_set)
                    if is_ok: #exist object
                        # pre_obj_center = center_box(latest_obj[4:])
                        # is_inside_roi_pre_obj = validate_center(pre_obj_center, False, roi_list)
                        if counts[cur_obj_id] > num_frame_ths and  count_in >= count_out and cur_obj_id not in already_count: # previous car lies inside roi
                            already_count.append(cur_obj_id)
                            max_movement_id = max(vote_movement[cur_obj_id], key=vote_movement[cur_obj_id].get)
                            tracking_mov[str(max_movement_id)] += 1
                            if tracking_info[index_box][0] == 1:
                                if max_movement_id not in num_1_out:
                                    num_1_out[max_movement_id] = 0
                                num_1_out[max_movement_id] += 1
                                num_object_out = num_1_out[max_movement_id]
                                class_type = "1@"
                            elif tracking_info[index_box][0] == 2:
                                if max_movement_id not in num_2_out:
                                    num_2_out[max_movement_id] = 0
                                num_2_out[max_movement_id] += 1
                                num_object_out = num_2_out[max_movement_id]
                                class_type = "2@"
                            elif tracking_info[index_box][0] == 3:
                                if max_movement_id not in num_3_out:
                                    num_3_out[max_movement_id] = 0
                                num_3_out[max_movement_id] += 1
                                num_object_out = num_3_out[max_movement_id]
                                class_type = "3@"
                            elif tracking_info[index_box][0] == 4:
                                if max_movement_id not in num_4_out:
                                    num_4_out[max_movement_id] = 0
                                num_4_out[max_movement_id] += 1
                                num_object_out = num_4_out[max_movement_id]
                                class_type = "4@"
                            results.append([fr_id, num_object_out, cur_center[0], cur_center[1], max_movement_id, class_type])
    print("Tracking {0} object. Count {1} object".format(len(total_object), len(np.array(results))))
    for key in tracking_mov:
        print(key, tracking_mov[key])
    if VISUALIZED:
        output = cv2.VideoWriter(os.path.join(PATH_VISUALIZE, cam_name + '.avi'), cv2.VideoWriter_fourcc('X','V','I','D'), 10.0, (width, height))
        idx = 0
        results = np.array(results)
        N = len(results)
        frame_id = results[:, 0].astype(np.int).reshape(N)
        text_summary = build_text_name_dict(moi_list)

        while (input.isOpened()):
            ret, frame = input.read()
            if not ret:
                break
            idx += 1
            indx_cur_fr = np.where(frame_id == idx)[0]
            annotate_fr = draw_roi(roi_list, frame)
            if len(indx_cur_fr)!=0:
                for result_id in indx_cur_fr:
                    cur_annotate = results[result_id] 
                    count_object = cur_annotate[1]
                    cv2.putText(annotate_fr, cur_annotate[-1]+"-"+str(count_object).zfill(5), (int(cur_annotate[2]), int(cur_annotate[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA) 
            annotate_fr = draw_roi(roi_list, frame)
            if len(indx_cur_fr)!=0:
                for result_id in indx_cur_fr:
                    cur_annotate = results[result_id] 
                    count_object = cur_annotate[1]
                    moi_id = cur_annotate[4]
                    class_type = cur_annotate[5]
                    text_summary[moi_id+"_"+class_type] = count_object
                    cv2.putText(annotate_fr, cur_annotate[4]+"-"+str(count_object).zfill(5), (int(cur_annotate[2]), int(cur_annotate[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA) 
            annotate_fr = draw_text_summarize(annotate_fr, text_summary, width, height)
            output.write(annotate_fr)
        input.release()
        output.release()
    np.save(os.path.join(PATH_RESULT, 'infox_' + cam_name + '.mp4'), results)
    return results

def find_movement(point,  obj_id, moi_list):
    # one vote for each time point lying inside MOI
    for moi_id in moi_list:
        moi_content = moi_list[moi_id]
        r, g, b, _ = moi_content[int(point[1])][int(point[0])]
        if r != 0 or g != 0 or b != 0:
            return moi_id, True
        #    print(obj_id, moi_id)
    return 0, False

def car_counting_one_shoot(cam_name, roi_list, moi_list, frame_offset_list):
    min_list = [330, 1195, 2030, 2895, 3760, 4625, 5490, 6355, 7220, 8085, 8950, 9815, 10680, 11545, 12410, 13275]   
    video_name = cam_name+".mp4"
    print("Processing", cam_name)
    tracking_info = np.load(os.path.join(PATH_TRACKING, 'info_' + video_name + '.npy'), allow_pickle = True)
    N = tracking_info.shape[0]
    frame_id = tracking_info[:, 1].astype(np.int).reshape(N)
    obj_id = tracking_info[:, 3].astype(np.int).reshape(N)
    results = []
    if len(sys.argv) > 2:
        VISUALIZED = True if sys.argv[2] == 'visualize' else False
    else:
        VISUALIZED = False
    if VISUALIZED:
        input = cv2.VideoCapture(os.path.join(PATH_VIDEO, cam_name + '.mp4'))
        height = int(input.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(input.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        width = 1280
        height = 720
    print("Using OSO")
    print("Visualizing ", cam_name)
    already_count = []
    vote_movement = {} # each key is obj_id, each value is A dictionary. Each key in A is movement_id, each value in A is count vote for that object to the corresponding 
    num_1_out = {} # each key is mv_id, each value count number of car 
    num_2_out = {} # each key is mv_id, each value count number of truck
    num_3_out = {} # each key is mv_id, each value count number of car 
    num_4_out = {} # each key is mv_id, each value count number of truck

    total_object = set()

    counter = {}
    tracking_mov = {'1': 0, '2': 0, '3':0, '4':0, '5':0, '6':0,'7':0, '8':0, '9':0, '10':0, '11':0,'12':0}
    for fr_id in range(1, max(frame_id)+1):
        index_cur_fr = np.where(frame_id==fr_id)[0]
        ok = False
        for ele_min_list in min_list: 
            if  (fr_id >= ele_min_list and fr_id <= ele_min_list + 520):
                ok = True
                break
        
        if ok:
            for index_box in index_cur_fr:
                cur_box = tracking_info[index_box][4:]
                cur_center = center_box(cur_box)
                cur_obj_id = tracking_info[index_box][3]
                total_object.add(cur_obj_id)

                move_id, inside_MOI = find_movement(cur_center, cur_obj_id, moi_list)
                if not inside_MOI:
                    continue
                else:
                    if cur_obj_id not in counter:
                        counter[cur_obj_id] = []
                    if cur_obj_id not in already_count: # previous car lies inside roi
                        #already_count.append(cur_obj_id)
                        #tracking_mov[str(move_id)] += 1
                        if tracking_info[index_box][0] == 1:
                            class_type = "1@"
                        elif tracking_info[index_box][0] == 2:
                            class_type = "2@"
                        elif tracking_info[index_box][0] == 3:
                            class_type = "3@"
                        elif tracking_info[index_box][0] == 4:
                            class_type = "4@"
                        if not VISUALIZED:
                            tmp_id = fr_id + frame_offset_list[int(move_id)]
                        else:
                            tmp_id = fr_id
                        counter[cur_obj_id] = [fr_id, 1, cur_center[0], cur_center[1], move_id, class_type]

    count_1, count_2, count_3, count_4 = 0, 0, 0, 0
    for key in counter:
        tracking_mov[counter[key][4]] += 1
        mov_id = counter[key][4]
        class_type = counter[key][-1]
        if class_type == "1@":
            if move_id not in num_1_out:
                num_1_out[move_id] = 0
            num_1_out[move_id] += 1
            num_obj_out = num_1_out[move_id]
            count_1 += 1
        if class_type == "2@":
            if move_id not in num_2_out:
                num_2_out[move_id] = 0
            num_2_out[move_id] += 1
            num_obj_out = num_2_out[move_id]
            count_2 += 1
        if class_type == "3@":
            if move_id not in num_3_out:
                num_3_out[move_id] = 0
            num_3_out[move_id] += 1
            num_obj_out = num_3_out[move_id]
            count_3 += 1
        if class_type == "4@":
            if move_id not in num_4_out:
                num_4_out[move_id] = 0
            num_4_out[move_id] += 1
            num_obj_out = num_4_out[move_id]
            count_4 += 1

        counter[key][1] = num_obj_out
        results.append(counter[key])
    print("Tracking {0} object. Capture at {1} frame. Count {2} object".format(len(total_object), len(np.array(results)), count_1 + count_2 + count_3 + count_4))
    for key in tracking_mov:
        if tracking_mov[key] > 0:
            print(key, tracking_mov[key])
    print("Type 1: {} | Type 2: {} | Type 3: {} | Type 4: {}".format(count_1, count_2, count_3, count_4))
    if VISUALIZED:
        output = cv2.VideoWriter(os.path.join(PATH_VISUALIZE, cam_name + '.avi'), cv2.VideoWriter_fourcc('X','V','I','D'), 10.0, (width, height))
        idx = 0
        results = np.array(results)
        N = len(results)
        frame_id = results[:, 0].astype(np.int).reshape(N)
        text_summary = build_text_name_dict(moi_list)
        total = 0

        while (input.isOpened()):
            ret, frame = input.read()
            if not ret:
                break
            idx += 1
            indx_cur_fr = np.where(frame_id == idx)[0]
            annotate_fr = draw_roi(roi_list, frame)
            if len(indx_cur_fr)!=0:
                for result_id in indx_cur_fr:
                    cur_annotate = results[result_id] 
                    count_object = cur_annotate[1]
                    cv2.putText(annotate_fr, cur_annotate[-1]+"-"+str(count_object), (int(cur_annotate[2]), int(cur_annotate[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA) 
            annotate_fr = draw_roi(roi_list, frame)
            if len(indx_cur_fr)!=0:
                for result_id in indx_cur_fr:
                    cur_annotate = results[result_id] 
                    count_object = cur_annotate[1]
                    moi_id = cur_annotate[4]
                    class_type = cur_annotate[5]
                    text_summary[moi_id+"_"+class_type] = count_object
                    cv2.putText(annotate_fr, cur_annotate[4]+"-"+str(count_object), (int(cur_annotate[2]), int(cur_annotate[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA) 
            annotate_fr = draw_text_summarize(annotate_fr, text_summary, width, height)
            output.write(annotate_fr)
        input.release()
        output.release()
    if not VISUALIZED:
        np.save(os.path.join(PATH_RESULT, 'info_' + cam_name + '.mp4'), results)
    return results

if __name__ == '__main__':
    roi_list = load_roi()
    moi_list = load_moi()
    frame_offset_list = load_frame_offset()
    with_off_set = ["cam_14", "cam_15"]
    one_shoot = ["cam_10", "cam_09","cam_12", "cam_20", "cam_01","cam_02","cam_03", "cam_04","cam_05","cam_06", "cam_07", "cam_08", "cam_11", "cam_13","cam_16", "cam_17", "cam_18", "cam_19", "cam_21", "cam_22", "cam_23", "cam_24","cam_25"]
    for video_name in os.listdir(PATH_VIDEO):
        if video_name.endswith(".mp4"):
            cam_name = video_name[:-4]
            if cam_name in one_shoot:
                result = car_counting_one_shoot(cam_name,roi_list[cam_name], moi_list[cam_name], frame_offset_list[cam_name])
            else:
                result = car_counting(cam_name, roi_list[cam_name], moi_list[cam_name], with_off_set)
