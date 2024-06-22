import os
import pyautogui
import pandas as pd
import time

# ----------------------------------------------- Path Configurations --------------------------------------------------------------
position_dataset = "/Users/charlotte/Library/CloudStorage/GoogleDrive-jialez@uchicago.edu/共享云端硬盘/Mining Mines/mine_lable/sample_full_ghana_part1.csv"
# position_dataset = None
store_dir = "/Users/charlotte/Library/CloudStorage/GoogleDrive-jialez@uchicago.edu/共享云端硬盘/Mining Mines/Satelite Image"


# --------------------------------------- Settings for automically collect images --------------------------------------------------
# please configure these according to your computer
search_bar_pos = (45, 45)
image_save_icon_pos = (540, 45)
pin_pos = (12, 120)

OPERTION_INTERVAL = 0.02     # the time interval between each operation, to garuantee the previous operation is finished.

target_label = "mine"

Google_earth_opened = False


def open_google_earth():
    '''
        Open the google earth application. (You should download it first on your computer)
    '''
    global Google_earth_opened
    os.system(r"open /Applications/Google\ Earth\ Pro.app")
    time.sleep(1)   # wait until application is launched.
    print("Google Earth Pro Opened.")
    Google_earth_opened = True


def search_coordinates(lat, lon):
    '''
        Click the search bar of google eartch and enter the position (latitude, longitude). 
    '''
    # click the search bar
    # hard code here, ** you need to set the coordinates yourself! Otherwise this script won't work **.
    pyautogui.click(x=search_bar_pos[0], y=search_bar_pos[1])  # the search bar is at this position.
    time.sleep(0.2)
    pyautogui.hotkey("command", "a")
    time.sleep(0.2)
    pyautogui.press("backspace")
    time.sleep(0.2)
    # search the longitude and latitude
    pyautogui.write("{}, {}".format(lat, lon), interval=OPERTION_INTERVAL)
    pyautogui.press("enter")
    time.sleep(0.5)

    pyautogui.click(x=pin_pos[0], y=pin_pos[1])
   


def save_image(image_path):
    '''
        Save the image according to the specific path.
    '''
    # save image
    pyautogui.click(x=image_save_icon_pos[0], y=image_save_icon_pos[1])   # click the save image icon
    print("Save Figure Icon Clicked. position: {}".format(pyautogui.position()))
    time.sleep(2)

    # enter the file name. 
    pyautogui.write(image_path, interval=OPERTION_INTERVAL)
    time.sleep(0.5)
    pyautogui.press("enter") 
    time.sleep(0.5)
    pyautogui.press("enter") 
    time.sleep(0.5)
    pyautogui.press("enter") 
    time.sleep(0.5)
    pyautogui.press("enter") 
    time.sleep(2)   # make sure the image has been saved.
    print("Image Saved: {}".format(image_path))

def collect_one_image(latitude, longitude):
    '''
        Given a specifed latitude and longitude:
        1. Seatch this location on Google Earth Pro.
        2. Download the iamge.
    '''

    global Google_earth_opened, target_label, store_dir

    file_name = "{}_{}_{}.jpg".format(target_label,latitude, longitude)

    image_store_path = os.path.join(store_dir, file_name)

    if (os.path.exists(image_store_path)):
        print("Image already collected: {}".format(image_store_path))
        return

    if not Google_earth_opened:
        open_google_earth()
    search_coordinates(latitude, longitude)
    save_image(file_name)


def main():

    # check whether the store path valid
    if not os.path.exists(store_dir):
        print("Store Path not exist: {}".format(store_dir))
        print("Please customize your store path")

    if position_dataset is not None:
        mine_positions = pd.read_csv(position_dataset)

        chunk_size = len(mine_positions) // 8

        # 分割DataFrame并分别保存为CSV
        for i in range(8):
            start = i * chunk_size
            # 确保最后一份包含所有剩余的行
            if i == 7:
                end = len(mine_positions)
            else:
                end = start + chunk_size
            # 保存每一份为CSV文件
            chunk_df = mine_positions.iloc[start:end]
            chunk_df.to_csv(f'sample_full_ghana_part{i+1}.csv', index=True)

        if target_label == "mine":
            mine_positions = mine_positions[mine_positions["Mining_Label"] == 1]
        else:
            mine_positions = mine_positions[mine_positions["Mining_Label"] == 0]

        # for _, row in mine_positions.iterrows():
        #     collect_one_image(row['Latitude'], row['Longitude'])
    else:s
        sample_latitude = 48.8587611
        sample_longitude = 2.2934333
        collect_one_image(sample_latitude, sample_longitude)


if __name__ == "__main__":
    main()
