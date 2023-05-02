train_image_path = "archive/train/"
test_image_path = "archive/val/"

id_map = {
    0: [0, 0, 0], # unlabelled
    1: [111, 74,  0], #static
    1: [ 81,  0, 81], #ground
    2: [128, 64,127], #road
    3: [244, 35,232], #sidewalk
    4: [250,170,160], #parking
    5: [230,150,140], #rail track
    6: [70, 70, 70], #building
    7: [102,102,156], #wall
    8: [190,153,153], #fence
    9: [180,165,180], #guard rail
    10: [150,100,100], #bridge
    11: [150,120, 90], #tunnel
    13: [153,153,153], #pole
    14: [153,153,153], #polegroup
    12: [250,170, 30], #traffic light
    13: [220,220,  0], #traffic sign
    17: [107,142, 35], #vegetation
    14: [152,251,152], #terrain
    15: [ 70,130,180], #sky
    16: [220, 20, 60], #person
    17: [255,  0,  0], #rider
    18: [  0,  0,142], #car
    19: [  0,  0, 70], #truck
    20: [  0, 60,100], #bus
    21: [  0,  0, 90], #caravan
    22: [  0,  0,110], #trailer
    23: [  0, 80,100], #train
    24: [  0,  0,230], #motorcycle
    25: [119, 11, 32], #bicycle
    30: [  0,  0,142] #license plate 
}