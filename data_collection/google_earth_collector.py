import pyautogui
import os
import time

# --------------------------------------- Settings for automically collect images --------------------------------------------------
# please configure these according to your computer
search_bar_pos = (40, 72)
image_save_icon_pos = (520, 70)
OPERTION_INTERVAL = 0.2     # the time interval between each operation, to garuantee the previous operation is finished.



def open_google_earth():
    os.system(r"open /Applications/Google\ Earth\ Pro.app")
    time.sleep(3)   # wait until application is launched.
    print("Google Earth Pro Opened.")

def search_coordinates(lat, lon):
    # click the search bar
    # hard code here, ** you need to set the coordinates yourself! Otherwise this script won't work **.
    pyautogui.click(x=search_bar_pos[0], y=search_bar_pos[1])  # the search bar is at this position.
    time.sleep(0.5)
    print("Search Bar Clicked. position: {}".format(pyautogui.position()))
    pyautogui.hotkey("command", "a")
    pyautogui.press("backspace")
    # search the longitude and latitude
    pyautogui.write("{}, {}".format(lat, lon), interval=OPERTION_INTERVAL)
    pyautogui.press("enter")
    time.sleep(1)  # wait until the search result is ready

def save_image(image_path):
    # save image
    pyautogui.click(x=image_save_icon_pos[0], y=image_save_icon_pos[1])   # click the save image icon
    print("Save Figure Icon Clicked. position: {}".format(pyautogui.position()))
    time.sleep(3)

    # enter the file name. 
    pyautogui.write(image_path, interval=OPERTION_INTERVAL)
    pyautogui.press("enter") 
    time.sleep(1)
    pyautogui.press("enter") 
    time.sleep(1)
    pyautogui.press("enter") 
    time.sleep(1)
    pyautogui.press("enter") 
    print("Image Saved: {}".format(image_path))

if __name__ == "__main__":
    sample_latitude = 41.5
    sample_longitude = 87.6

    store_dir = "/Users/peiranqin/Desktop/African-mining/image_dataset/not_mine"
    file_name = "{}_{}.png".format(sample_latitude, sample_longitude)

    image_store_path = os.path.join(store_dir, file_name)

    open_google_earth()
    search_coordinates(sample_latitude, sample_longitude)
    save_image(image_store_path)
