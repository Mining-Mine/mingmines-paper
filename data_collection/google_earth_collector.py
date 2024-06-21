import os
import pyautogui
import pandas as pd
import time

# ----------------------------------------------- Path Configurations --------------------------------------------------------------
position_dataset = "/Users/peiranqin/Desktop/African-mining/image_dataset/sample_not_mine.xlsx"
# position_dataset = None
store_dir = "/Users/peiranqin/Desktop/African-mining/image_dataset/not_mine"


# --------------------------------------- Settings for automically collect images --------------------------------------------------
# please configure these according to your computer
search_bar_pos = (40, 72)
image_save_icon_pos = (510, 70)
pin_pos = (12, 162)
OPERTION_INTERVAL = 0.01     # the time interval between each operation, to garuantee the previous operation is finished.


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
    time.sleep(1)

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
    time.sleep(3)   # make sure the image has been saved.
    print("Image Saved: {}".format(image_path))

def collect_one_image(latitude, longitude):
    '''
        Given a specifed latitude and longitude:
        1. Seatch this location on Google Earth Pro.
        2. Download the iamge.
    '''
    global Google_earth_opened
    file_name = "{}_{}.jpg".format(latitude, longitude)

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
        mine_not_mine_positions = pd.read_excel(position_dataset, engine='openpyxl')
        for _, row in mine_not_mine_positions.iterrows():
            collect_one_image(row['Latitude'], row['Longitude'])
    else:
        sample_latitude = 48.8587611
        sample_longitude = 2.2934333
        collect_one_image(sample_latitude, sample_longitude)


if __name__ == "__main__":
    main()
