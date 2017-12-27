from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as pltdate
from PIL import Image, ImageDraw
import pandas as pd

dates = [
"Sat  9 Dec 05:59:13 CEST 2017",
"Sat  9 Dec 06:12:30 CEST 2017",
"Sat  9 Dec 06:40:59 CEST 2017",
"Sat  9 Dec 06:52:31 CEST 2017",
"Sat  9 Dec 06:59:23 CEST 2017",
"Sat  9 Dec 06:57:30 CEST 2017",
"Sat  9 Dec 06:59:11 CEST 2017",
"Sat  9 Dec 07:02:26 CEST 2017",
"Sat  9 Dec 07:08:34 CEST 2017"
]

dates = pd.to_datetime(dates)

values = [0,1,2,3,1,1,2,1,0]
data = [str(i) for i in values]



#generate image
WIDTH, HEIGHT = 4000, 400
BORDER = 70
W = WIDTH - (2 * BORDER)
H = HEIGHT - (2 * BORDER)


colors = { '0': "lime", '1' : (255,200,200), '2' : (255,100,100), '3' : (255,0,0) }

image = Image.new("RGB", (WIDTH, HEIGHT), "white")
min_date = dates[0]
max_date = datetime.now()
#print(min_date)
#print(max_date)
interval = max_date - min_date
#print(interval.days)

#draw frame
draw = ImageDraw.Draw(image)
draw.rectangle((BORDER, BORDER, WIDTH-BORDER, HEIGHT-BORDER), fill=(128,128,128), outline=(0,0,0))

#draw circles
circle_w = 10
range_secs = W / interval.total_seconds()
#print(range_secs)
for i in range(len(dates)):
    wat = dates[i] - min_date
    offset_sec = (dates[i] - min_date).total_seconds()
    offset = range_secs * offset_sec
    x = BORDER + offset
    draw.ellipse((x, BORDER + 50, x + circle_w, BORDER + 50 + circle_w), outline=colors[data[i]])
    #draw.text((x, BORDER + 75), str(i), fill=colors[data[i]])

#draw rectangles
range_days = W / (interval.days + 1)
#print("range_days",range_days)
current_date = min_date
date_month = min_date + relativedelta(months=1)
current_index = 0
for i in range(interval.days + 1):
    max_color = '0'
    while dates[current_index].date() == current_date.date():
        if int(data[current_index]) > int(max_color):
            max_color = data[current_index]
        current_index += 1
        if current_index > len(dates) - 1:
            current_index = 0
    x = BORDER + range_days * i
    draw.rectangle((x, BORDER + 100, x+range_days, BORDER + 100 + 50), fill=colors[max_color], outline=(0,0,0))
    if current_date == date_month:
        draw.line((x, BORDER + 100 +50, x, H + BORDER + 20), fill="black")
        draw.text((x, H + BORDER + 20), str(date_month.date()), fill="black")
        date_month = date_month + relativedelta(months=1)
    #draw.text((x, BORDER + 175), str(i), fill=colors[max_color])
    current_date = current_date + timedelta(days=1)

#draw start and end dates
draw.text((BORDER, H + BORDER + 20), str(min_date.date()), fill="black")
draw.text((BORDER + W, H + BORDER + 20), str(max_date.date()), fill="black")

image.save("date.png")




