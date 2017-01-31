# carnd-project-3
before region masking it appears that my NN might have been using pieces of the surrounding environment to "drive"
the tricky part about having a ton of data was that it knows the course mainly goes left, and then always thinks it should turn left (but this was also done before i put region masking in) this also occurred when i had driven a lot of laps around the course without many corrective action clips. those corrective clips have now been the basis of my working model. the whole process trained better when i did small sections at a time. take whatever worked and then when it failed, train a specific failure. as my samples size got larger, i needed to repeat the desired behaviour several times in order for it to make an impact. when doing it this way, it seems like the car was able to "drive" more intelligently, instead of trying to memorize what i did

i had a little bit of trouble with the bridge because it had black barriers, where everything else has brighter, almost white barriers. once the car was on the bridge it would easily go straight, probably because the sides were so clear to distinguish.

parsing false positive might be relevant next time too. i know with recording sometimes it's hard to time the samples correctly

I also should have probably used the other cameras, I can imagine they help a lot when the network is trying to recognize the boundaries that exist and what to stay away from.

i've notice that there are certain portions of the track that mess up other portions, specifically right before the bridge messes up the very beggining of the course when you start, and vice versa. apparently the lighting on the left bank is very similar and my car sometimes changes it's behaviour on one piece of the track when i'm really just trying to alter the other piece.

additional sensors to figure out where the ground is vs the color of the ground, would definitely help too.