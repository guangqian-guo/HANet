################add by guogq#####################


#用来在图片上标注框 用于tinyperson数据集
import os
import json
from PIL import Image,ImageDraw
import PIL.ImageFont as ImageFont

#wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
  #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
wordname_2 = ['sea_person','earth_person']
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

root_dir = os.getcwd()
print(root_dir)
gt_file = '/home/ubuntu/Guo/TOV_mmdetection-main/data/tiny_set/annotations/task/tiny_set_test_all.json'
# json_file = os.path.join(root_dir,"figure/bbox.json")
# gt_file = os.path.join(root_dir,"annotations/tiny_set_test.json")
# gt_file = os.path.join(root_dir,"bbox_merge_nms0.5.json")
# img_file = os.path.join(root_dir,"test\labeled_images")
img_file = '/home/ubuntu/Guo/TOV_mmdetection-main/work-dir/Tinyperson/centernet-r50/results+5/'
# img_file = '/home/ubuntu/Guo/TOV_mmdetection-main/data/tiny_set/test/labeled_images/'
i=0

image_list = os.listdir(img_file)
# image_list = ['bb_V0009_I0002800.jpg']
print(image_list)
bbox = []
gt_box = []
img_id_list = []
with open(gt_file, encoding='utf-8') as f:
	d = json.load(f)
	annotations = d['annotations']
	images = d["images"]
	# images= d["images"]

	for n, img_name in enumerate(image_list):
		# print(image_id_)
		gt_box_ = []
		# img_name = img_name.split("/")[1]
		print(img_name)
		for i in images:
			
			if i["file_name"].split("/")[1] == img_name:
				img_id = i["id"]
				img_id_list.append(img_id)
				break

		for a in annotations:
			# print(i)
			# print(m['image_id'])
			if a['image_id'] == img_id:
				a['bbox'].append(a['category_id'])
				gt_box_.append(a['bbox'])
				# ge_box_.append(a['bbox'])
			
		gt_box.append(gt_box_)
		f.close()

# with open(json_file,encoding='utf-8') as f:
# 	d = json.load(f)
# 	for n,image_id_ in enumerate(img_id_list):
# 		box = []
# 		for m in d:
# 			# print(m['image_id'])
# 			if m['image_id'] == image_id_:
# 				m['bbox'].append(m['category_id'])
# 				box.append(m['bbox'])

			
# 		bbox.append(box)
# 		f.close()

# print(gt_box)


# save_path = '/home/ubuntu/Guo/TOV_mmdetection-main/data/tiny_set/test/vis_images/'
save_path = img_file
for j,image in enumerate(image_list):
	img = Image.open(os.path.join(img_file,image))
	width,height = img.size
	img = img.resize((int(width/2),int(height/2)))

	draw = ImageDraw.Draw(img)
	length = int(len(gt_box[j])*1)
	# font = ImageFont.truetype('arial.ttf', 30)
		
	for n,box in enumerate(gt_box[j][:length]):

		##predictions
		
		# box = bbox[j][n]
		#
		# x1 = box[0]
		# y1 = box[1]
		# x2 = box[0]+box[2]
		# y2 = box[1]+box[3]
		# draw.rectangle((x1, y1, x2, y2), outline='yellow')
		#draw.text((x1,y1),text=wordname_15[box[4]-1],fill='red')
		## GT
		box = gt_box[j][n]
		w = box[2]
		h = box[3]
		size = w * h
		
		x1 = box[0]
		y1 = box[1]
		x2 = box[0] + box[2]
		y2 = box[1] + box[3]
		
		draw.rectangle((x1/2, y1/2, x2/2, y2/2), outline='Purple',width=1)
		# draw.rectangle((x1-15, y1-10, x2-8, y2-7), outline='Cyan', width=3)
		# draw.text((x1, y1-40), text='IOU=0.59', fill='yellow',font=font)
	img.save(save_path+image)  # save picture


		# print(bbox)
	
	# print(len(annotations))  #25288
	
		# while True:
		# 	line = f.readline()
		# 	if line:
		# 		d = json.loads(line)
		# 		annotations = d['annotations']
		#
		# 	else:
		# 		break
			
