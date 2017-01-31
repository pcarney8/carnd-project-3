# carnd-project-3

## Architecture
I originally began with my own architecture from the Keras lab, and immediately ran into several issues. After some trial and error I decided to use Comma.ai's model as a starting point intstead. I read on the forums how people had used comma.ai's and NVIDIA's models, so I gave them both a look and pieced together the two. I have a Lambda in the beginning, 3 Convultional layers, then I removed the ELU's from comma.ai's model and replaced them with relu's. This seems to work fairly well and was a litle more similar to NVIDIA's model. There wasn't a huge difference in validation loss between the two. I tried adding a few more fully connected layers at the bottom and that too didn't have a significant impact.

I used an Adam optimizer so I wouldn't have fool around with the learning rate as much, and I kept the dropout layers in to make sure I reduced overfitting.

I also found that 5 epochs was more than enough, my validation loss hovered around 0.017 and never got much lower no matter how many epochs.

## Solution Design
I originally started by reducing the size of the image to 32x16 so my laptop could handle that kind of workload. I used a generator, but I didn't load from disk in the generator, it 

Before region masking it appears that my CNN might have been using pieces of the surrounding environment to "drive". I originally took off only the top 20 pixels for the region mask, and thought that would be sufficient. It was not. My CNN was still taking cues from the last little bit of the top to try and predict the steering angle. I took it down another 10 pixels, and viola, my CNN got around the track first try.

I also flipped the image and reversed the steering angle. This doubled my data which was very helpful. Granted I spent so much time training and re-training before I brought in the region mask, it probably would have been better to just have figured out how to use the right and left cameras.

## Training Process
The tricky part about having a ton of data was that it knows the course mainly goes left, and then always thinks it should turn left, but this was happening before I put region masking in. This also occurred when I had driven a lot of laps around the course without many corrective action clips. Those corrective clips have now been the basis of my working model. The whole process trained better when I did small sections at a time. Take whatever worked and then when it failed, train a specific failure. As my samples size got larger, I needed to repeat the desired behaviour at least a couple of times in order for it to make an impact. When doing it this way, it seems like the car was able to "drive" more intelligently, instead of trying to memorize what I did.

I had a little bit of trouble with the area right before the bridge because it had black barriers, where everything else has brighter, almost white barriers. Once the car was on the bridge it would easily go straight, probably because the sides were so clear and easy to distinguish.

# Learnings
Region Masking! Always use it when you can. There's so much irrelevant information at the top of the picture.

Parsing false positive might be relevant next time. I know with recording sometimes it's hard to time the samples correctly. I also should have probably used the other cameras, I can imagine they help a lot when the network is trying to recognize the boundaries that exist and what to stay away from.

I noticed that there are certain portions of the track that mess up other portions, specifically right before the bridge messes up the very beginning of the course when you start, and vice versa. apparently the lighting on the left bank is very similar and my car sometimes changes it's behaviour on one piece of the track when I'm really just trying to alter the other piece. Although this also might have been because I wasn't using region masking yet!

I think additional sensors to figure out where the ground is vs the color of the ground, would definitely help (LIDAR???).

There's been a delicate balance of getting the car to turn correctly, based on the lines of the road vs the car following directly on the lines of the road. I could tell that my model needed to be "smoothed out" when it would follow on top of the lines, all I needed was to drive in the middle more. But occasionally if I drove in the middle too much, the car would stop turning as quickly when it came near the lines. But again, before region masking!