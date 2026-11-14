import numpy as np, sys
W,H,FPS,N=160,120,30,30*95
out=open('fake.y4m','wb'); out.write(f'YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420jpeg\n'.encode())
for i in range(N):
    t=i/FPS; ph=(1.2*t)%1.0
    pulse=np.exp(-((ph-0.15)/0.08)**2)*1.0+0.35*np.exp(-((ph-0.45)/0.1)**2)
    y=int(110+ 14*(pulse-0.3))
    out.write(b'FRAME\n'); out.write(bytes([y])*(W*H)); out.write(bytes([100])*(W*H//4)); out.write(bytes([190])*(W*H//4))
out.close()
