import numpy as np

import nibabel as nib
import json
import cv2


def mask2json(pre_target, file_info):
    mask = pre_target['mask']
    landmark = pre_target['landmark']

    # get contours
    contours = {1: None, 2: None }
    for i in range(1, 3):
        binary = np.zeros_like(mask)
        binary[mask == i] = 255
        contour, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        # 检测的轮廓中有很多，提取最长的轮廓，为上颌骨轮廓
        if len(contour)==0:
            contours[i] = []
        else:
            temp = contour[0]
            for j in contour[1:]:
                if len(j) > len(temp):
                    temp = j
            contours[i] = temp.squeeze()
    FileInfo_Name = file_info['name']
    FileName = file_info['name']
    FileInfo_Name += '.jpg' if file_info['jpg_JPG'] == 'jpg' else '.JPG'
    FileName += '_jpg' if file_info['jpg_JPG'] == 'jpg' else '_JPG'

    # 将两个区域转换为nii.gz 格式
    my_arr = np.zeros_like(mask)
    my_arr[mask == 1] = 4
    my_arr[mask == 2] = 5
    my_arr = np.transpose(my_arr)
    new_image = nib.Nifti1Image(my_arr, np.eye(4))
    # new_image.set_data_dtype(np.my_dtype)
    nib.save(new_image, './data/test/' + FileName + '_Label.nii.gz')

    json_data = {}
    json_data['Config'] = {
        "Ambient": 0.3,
        "CanRepeatLabel": False,
        "Diffuse": 0.6,
        "ExcludeFileSuffixList": "",
        "FileTagTitle1": "",
        "FileTagTitle2": "",
        "FileTagTitle3": "",
        "IsLoadDicomToVideo": False,
        "IsSaveSrcFile": True,
        "KEY": "28YvHZBdojwr2QABlcHjtAvCsgWOXAWEUq1lj8Lee8k=",
        "LabelSavePath": "",
        "LandMark3DScale": 1.0,
        "LandMarkActorScale": 1.0,
        "PlayInterval": 100,
        "PolyPointScale": 1.0,
        "Record_Time": 5,
        "RectActorScale": 1.0,
        "SegModelPaths": None,
        "SliceCount": 1,
        "SliceSpacing": 1.0,
        "Specular": 0.1,
        "zh_CN": {
            "Ambient": "Ambient",
            "CanRepeatLabel": "Repeat Landmark",
            "Diffuse": "Diffuse",
            "ExcludeFileSuffixList": "Filter out",
            "FileTagTitle1": "Parent Tag Name",
            "FileTagTitle2": "Additional Tag Name",
            "FileTagTitle3": "Child Tag Name",
            "IsLoadDicomToVideo": "Load DICOM as Video",
            "IsSaveSrcFile": "Save Source",
            "KEY": "Key Code",
            "KEY_Type": "Encryption Type",
            "LandMark3DScale": "3D Sphere Size",
            "LandMarkActorScale": "Landmark Size",
            "PlayInterval": "Play Interval",
            "PolyPointScale": "Contour Point Size",
            "Record_Time": "Screen Recording (sec)",
            "RectActorScale": "Box Point Size",
            "SliceCount": "Slice Number",
            "SliceSpacing": "Slice Spacing (mm)",
            "Specular": "Specular"
        }
    }
    json_data['CurvePointIds'] = None
    json_data['Curves'] = [
        {
            "Shapes": None,
            "SliceType": 3
        }
    ]
    json_data['FileInfo'] = {
        "Commit": "29fb0ab",
        "Data": "",
        "Depth": 3,
        "Height": file_info['height'],
        "Name": FileInfo_Name,
        "Version": "v2.6.0",
        "Width": file_info['width']
    }
    json_data['FileName'] = FileName
    json_data['LabelGroup'] = [
        {
            "Childs": None,
            "ID": 0
        },
        {
            "Childs": [
                4,
                5
            ],
            "ID": 1,
            "Type": "Polygon"
        },
        {
            "Childs": [
                6,
                7
            ],
            "ID": 2,
            "Type": "DrawLine"
        },
        {
            "Childs": [
                8,
                9,
                10,
                11,
                12,
                13
            ],
            "ID": 3,
            "Type": "Landmark"
        }
    ]
    json_data['Models'] = {
        "AngleModel": None,
        "BoundingBox3DLabelModel": None,
        "BoundingBoxLabelModel": None,
        "CircleModel": None,
        "ColorLabelTableModel": [
            {
                "Color": [
                    41,
                    35,
                    190,
                    255
                ],
                "Desc": "分割",
                "ID": 1
            },
            {
                "Color": [
                    132,
                    225,
                    108,
                    255
                ],
                "Desc": "曲线",
                "ID": 2
            },
            {
                "Color": [
                    214,
                    174,
                    82,
                    255
                ],
                "Desc": "关键点",
                "ID": 3
            },
            {
                "Color": [
                    144,
                    73,
                    241,
                    255
                ],
                "Desc": "upperJaw",
                "ID": 4
            },
            {
                "Color": [
                    241,
                    187,
                    233,
                    255
                ],
                "Desc": "underJaw",
                "ID": 5
            },
            {
                "Color": [
                    255,
                    0,
                    127,
                    255
                ],
                "Desc": "nasionCurve",
                "ID": 6
            },
            {
                "Color": [
                    255,
                    255,
                    0,
                    255
                ],
                "Desc": "frontalBoneCurve",
                "ID": 7
            },
            {
                "Color": [
                    12,
                    62,
                    153,
                    255
                ],
                "Desc": "upperLip",
                "ID": 8
            },
            {
                "Color": [
                    36,
                    94,
                    13,
                    255
                ],
                "Desc": "underLip",
                "ID": 9
            },
            {
                "Color": [
                    28,
                    6,
                    183,
                    255
                ],
                "Desc": "upperMidpoint",
                "ID": 10
            },
            {
                "Color": [
                    71,
                    222,
                    179,
                    255
                ],
                "Desc": "underMidpoint",
                "ID": 11
            },
            {
                "Color": [
                    18,
                    77,
                    200,
                    255
                ],
                "Desc": "chin",
                "ID": 12
            },
            {
                "Color": [
                    67,
                    187,
                    139,
                    255
                ],
                "Desc": "nasion",
                "ID": 13
            }
        ],
        "EllipseModel": None,
        "FrameLabelModel": None,
        "LabelDetailModel": None,
        "LandMarkListModel": {
            "Lines": None,
            "Points": [
                {
                    "ImageIndex": 0,
                    "LabelList": [
                        {
                            "Label": 8,
                            "Position": [
                                landmark[8][0],
                                landmark[8][1],
                                0.0
                            ]
                        },
                        {
                            "Label": 13,
                            "Position": [
                                landmark[13][0],
                                landmark[13][1],
                                0.0
                            ]
                        },
                        {
                            "Label": 12,
                            "Position": [
                                landmark[12][0],
                                landmark[12][1],
                                0.0
                            ]
                        },
                        {
                            "Label": 11,
                            "Position": [
                                landmark[11][0],
                                landmark[11][1],
                                0.0
                            ]
                        },
                        {
                            "Label": 9,
                            "Position": [
                                landmark[9][0],
                                landmark[9][1],
                                0.0
                            ]
                        },
                        {
                            "Label": 10,
                            "Position": [
                                landmark[10][0],
                                landmark[10][1],
                                0.0
                            ]
                        }
                    ],
                    "ViewType": 3
                }
            ]
        },
        "MaskEditRecordModel": None,
        "MeasureModel": None,
        "MeshModel": None,
        "MprPositionModel": None,
        "PolygonModel": {
            "3": [
                0
            ]
        },
        "PolygonModel2": None,
        "ResliceLabelModel": None
    }
    json_data['Polys'] = [
        {
            "Shapes": [
                {
                    "FindInPolygonVal": 1,
                    "ImageFrame": 0,
                    "Points": [{"Pos": [float(x), float(y), 0.0]} for x, y in contours[1]],
                    "SliceType": 3,
                    "labelType": 4
                },
                {
                    "FindInPolygonVal": 1,
                    "ImageFrame": 0,
                    "Points": [{"Pos": [float(x), float(y), 0.0]} for x, y in contours[2]],
                    "SliceType": 3,
                    "labelType": 5
                }
            ],
            "SliceType": 3
        }
    ]
    json_data['Timeline'] = None

    with open('./data/test/' + FileName + "_Label.json", "w") as f:
        json.dump(json_data, f)
