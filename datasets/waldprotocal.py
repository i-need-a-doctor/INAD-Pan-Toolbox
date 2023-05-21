import cv2

def WaldProtocal(hhr_pan, hr_ms):
    hr_pan = cv2.resize(hhr_pan, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    lr_ms = cv2.resize(cv2.GaussianBlur(hr_ms, (9, 9), 0), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    return hr_pan, lr_ms
