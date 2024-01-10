import os
import time

packed_mask = [
    [0x1, 0x0, 0x0, 0x0],
    [0x11, 0x0, 0x0, 0x0],
    [0x111, 0x0, 0x0, 0x0],
    [0x1111, 0x0, 0x0, 0x0],
    [0x11111, 0x0, 0x0, 0x0],
    [0x111111, 0x0, 0x0, 0x0],
    [0x1111111, 0x0, 0x0, 0x0],
    [0x11111111, 0x0, 0x0, 0x0],
    [0x11111111, 0x1, 0x0, 0x0],
    [0x11111111, 0x11, 0x0, 0x0],
    [0x11111111, 0x111, 0x0, 0x0],
    [0x11111111, 0x1111, 0x0, 0x0],
    [0x11111111, 0x11111, 0x0, 0x0],
    [0x11111111, 0x111111, 0x0, 0x0],
    [0x11111111, 0x1111111, 0x0, 0x0],
    [0x11111111, 0x11111111, 0x0, 0x0],
    [0x11111111, 0x11111111, 0x1, 0x0],
    [0x11111111, 0x11111111, 0x11, 0x0],
    [0x11111111, 0x11111111, 0x111, 0x0],
    [0x11111111, 0x11111111, 0x1111, 0x0],
    [0x11111111, 0x11111111, 0x11111, 0x0],
    [0x11111111, 0x11111111, 0x111111, 0x0],
    [0x11111111, 0x11111111, 0x1111111, 0x0],
    [0x11111111, 0x11111111, 0x11111111, 0x0],
    [0x11111111, 0x11111111, 0x11111111, 0x1],
    [0x11111111, 0x11111111, 0x11111111, 0x11],
    [0x11111111, 0x11111111, 0x11111111, 0x111],
    [0x11111111, 0x11111111, 0x11111111, 0x1111],
    [0x11111111, 0x11111111, 0x11111111, 0x11111],
    [0x11111111, 0x11111111, 0x11111111, 0x111111],
    [0x11111113, 0x11111111, 0x11111111, 0x111111],
    [0x11111133, 0x11111111, 0x11111111, 0x111111],
    [0x11111333, 0x11111111, 0x11111111, 0x111111],
    [0x11113333, 0x11111111, 0x11111111, 0x111111],
    [0x11133333, 0x11111111, 0x11111111, 0x111111],
    [0x11333333, 0x11111111, 0x11111111, 0x111111],
    [0x13333333, 0x11111111, 0x11111111, 0x111111],
    [0x33333333, 0x11111111, 0x11111111, 0x111111],
    [0x33333333, 0x11111113, 0x11111111, 0x111111],
    [0x33333333, 0x11111133, 0x11111111, 0x111111],
    [0x33333333, 0x11111333, 0x11111111, 0x111111],
    [0x33333333, 0x11113333, 0x11111111, 0x111111],
    [0x33333333, 0x11133333, 0x11111111, 0x111111],
    [0x33333333, 0x11333333, 0x11111111, 0x111111],
    [0x33333333, 0x13333333, 0x11111111, 0x111111],
    [0x33333333, 0x33333333, 0x11111111, 0x111111],
    [0x33333333, 0x33333333, 0x11111113, 0x111111],
    [0x33333333, 0x33333333, 0x11111133, 0x111111],
    [0x33333333, 0x33333333, 0x11111333, 0x111111],
    [0x33333333, 0x33333333, 0x11113333, 0x111111],
    [0x33333333, 0x33333333, 0x11133333, 0x111111],
    [0x33333333, 0x33333333, 0x11333333, 0x111111],
    [0x33333333, 0x33333333, 0x13333333, 0x111111],
    [0x33333333, 0x33333333, 0x33333333, 0x111111],
    [0x33333333, 0x33333333, 0x33333333, 0x111113],
    [0x33333333, 0x33333333, 0x33333333, 0x111133],
    [0x33333333, 0x33333333, 0x33333333, 0x111333],
    [0x33333333, 0x33333333, 0x33333333, 0x113333],
    [0x33333333, 0x33333333, 0x33333333, 0x133333],
    [0x33333333, 0x33333333, 0x33333333, 0x333333],
    [0x33333337, 0x33333333, 0x33333333, 0x333333],
    [0x33333377, 0x33333333, 0x33333333, 0x333333],
    [0x33333777, 0x33333333, 0x33333333, 0x333333],
    [0x33337777, 0x33333333, 0x33333333, 0x333333],
    [0x33377777, 0x33333333, 0x33333333, 0x333333],
    [0x33777777, 0x33333333, 0x33333333, 0x333333],
    [0x37777777, 0x33333333, 0x33333333, 0x333333],
    [0x77777777, 0x33333333, 0x33333333, 0x333333],
    [0x77777777, 0x33333337, 0x33333333, 0x333333],
    [0x77777777, 0x33333377, 0x33333333, 0x333333],
    [0x77777777, 0x33333777, 0x33333333, 0x333333],
    [0x77777777, 0x33337777, 0x33333333, 0x333333],
    [0x77777777, 0x33377777, 0x33333333, 0x333333],
    [0x77777777, 0x33777777, 0x33333333, 0x333333],
    [0x77777777, 0x37777777, 0x33333333, 0x333333],
    [0x77777777, 0x77777777, 0x33333333, 0x333333],
    [0x77777777, 0x77777777, 0x33333337, 0x333333],
    [0x77777777, 0x77777777, 0x33333377, 0x333333],
    [0x77777777, 0x77777777, 0x33333777, 0x333333],
    [0x77777777, 0x77777777, 0x33337777, 0x333333],
    [0x77777777, 0x77777777, 0x33377777, 0x333333],
    [0x77777777, 0x77777777, 0x33777777, 0x333333],
    [0x77777777, 0x77777777, 0x37777777, 0x333333],
    [0x77777777, 0x77777777, 0x77777777, 0x333333],
    [0x77777777, 0x77777777, 0x77777777, 0x333337],
    [0x77777777, 0x77777777, 0x77777777, 0x333377],
    [0x77777777, 0x77777777, 0x77777777, 0x333777],
    [0x77777777, 0x77777777, 0x77777777, 0x337777],
    [0x77777777, 0x77777777, 0x77777777, 0x377777],
    [0x77777777, 0x77777777, 0x77777777, 0x777777],
    [0x7777777f, 0x77777777, 0x77777777, 0x777777],
    [0x777777ff, 0x77777777, 0x77777777, 0x777777],
    [0x77777fff, 0x77777777, 0x77777777, 0x777777],
    [0x7777ffff, 0x77777777, 0x77777777, 0x777777],
    [0x777fffff, 0x77777777, 0x77777777, 0x777777],
    [0x77ffffff, 0x77777777, 0x77777777, 0x777777],
    [0x7fffffff, 0x77777777, 0x77777777, 0x777777],
    [0xffffffff, 0x77777777, 0x77777777, 0x777777],
    [0xffffffff, 0x7777777f, 0x77777777, 0x777777],
    [0xffffffff, 0x777777ff, 0x77777777, 0x777777],
    [0xffffffff, 0x77777fff, 0x77777777, 0x777777],
    [0xffffffff, 0x7777ffff, 0x77777777, 0x777777],
    [0xffffffff, 0x777fffff, 0x77777777, 0x777777],
    [0xffffffff, 0x77ffffff, 0x77777777, 0x777777],
    [0xffffffff, 0x7fffffff, 0x77777777, 0x777777],
    [0xffffffff, 0xffffffff, 0x77777777, 0x777777],
    [0xffffffff, 0xffffffff, 0x7777777f, 0x777777],
    [0xffffffff, 0xffffffff, 0x777777ff, 0x777777],
    [0xffffffff, 0xffffffff, 0x77777fff, 0x777777],
    [0xffffffff, 0xffffffff, 0x7777ffff, 0x777777],
    [0xffffffff, 0xffffffff, 0x777fffff, 0x777777],
    [0xffffffff, 0xffffffff, 0x77ffffff, 0x777777],
    [0xffffffff, 0xffffffff, 0x7fffffff, 0x777777],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x777777],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x77777f],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x7777ff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x777fff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x77ffff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x7fffff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0xffffff]
]

'''
# distributed_mask = [
#     [0x1, 0x0, 0x0, 0x0],
#     [0x3, 0x0, 0x0, 0x0],
#     [0x7, 0x0, 0x0, 0x0],
#     [0xf, 0x0, 0x0, 0x0],
#     [0x1f, 0x0, 0x0, 0x0],
#     [0x3f, 0x0, 0x0, 0x0],
#     [0x7f, 0x0, 0x0, 0x0],
#     [0xff, 0x0, 0x0, 0x0],
#     [0x1ff, 0x0, 0x0, 0x0],
#     [0x3ff, 0x0, 0x0, 0x0],
#     [0x7ff, 0x0, 0x0, 0x0],
#     [0xfff, 0x0, 0x0, 0x0],
#     [0x1fff, 0x0, 0x0, 0x0],
#     [0x3fff, 0x0, 0x0, 0x0],
#     [0x7fff, 0x0, 0x0, 0x0],
#     [0xffff, 0x0, 0x0, 0x0],
#     [0x1ffff, 0x0, 0x0, 0x0],
#     [0x3ffff, 0x0, 0x0, 0x0],
#     [0x7ffff, 0x0, 0x0, 0x0],
#     [0xfffff, 0x0, 0x0, 0x0],
#     [0x1fffff, 0x0, 0x0, 0x0],
#     [0x3fffff, 0x0, 0x0, 0x0],
#     [0x7fffff, 0x0, 0x0, 0x0],
#     [0xffffff, 0x0, 0x0, 0x0],
#     [0x1ffffff, 0x0, 0x0, 0x0],
#     [0x3ffffff, 0x0, 0x0, 0x0],
#     [0x7ffffff, 0x0, 0x0, 0x0],
#     [0xfffffff, 0x0, 0x0, 0x0],
#     [0x1fffffff, 0x0, 0x0, 0x0],
#     [0x3fffffff, 0x0, 0x0, 0x0],
#     [0x7fffffff, 0x0, 0x0, 0x0],
#     [0xffffffff, 0x0, 0x0, 0x0],
#     [0xffffffff, 0x1, 0x0, 0x0],
#     [0xffffffff, 0x3, 0x0, 0x0],
#     [0xffffffff, 0x7, 0x0, 0x0],
#     [0xffffffff, 0xf, 0x0, 0x0],
#     [0xffffffff, 0x1f, 0x0, 0x0],
#     [0xffffffff, 0x3f, 0x0, 0x0],
#     [0xffffffff, 0x7f, 0x0, 0x0],
#     [0xffffffff, 0xff, 0x0, 0x0],
#     [0xffffffff, 0x1ff, 0x0, 0x0],
#     [0xffffffff, 0x3ff, 0x0, 0x0],
#     [0xffffffff, 0x7ff, 0x0, 0x0],
#     [0xffffffff, 0xfff, 0x0, 0x0],
#     [0xffffffff, 0x1fff, 0x0, 0x0],
#     [0xffffffff, 0x3fff, 0x0, 0x0],
#     [0xffffffff, 0x7fff, 0x0, 0x0],
#     [0xffffffff, 0xffff, 0x0, 0x0],
#     [0xffffffff, 0x1ffff, 0x0, 0x0],
#     [0xffffffff, 0x3ffff, 0x0, 0x0],
#     [0xffffffff, 0x7ffff, 0x0, 0x0],
#     [0xffffffff, 0xfffff, 0x0, 0x0],
#     [0xffffffff, 0x1fffff, 0x0, 0x0],
#     [0xffffffff, 0x3fffff, 0x0, 0x0],
#     [0xffffffff, 0x7fffff, 0x0, 0x0],
#     [0xffffffff, 0xffffff, 0x0, 0x0],
#     [0xffffffff, 0x1ffffff, 0x0, 0x0],
#     [0xffffffff, 0x3ffffff, 0x0, 0x0],
#     [0xffffffff, 0x7ffffff, 0x0, 0x0],
#     [0xffffffff, 0xfffffff, 0x0, 0x0],
#     [0xffffffff, 0x1fffffff, 0x0, 0x0],
#     [0xffffffff, 0x3fffffff, 0x0, 0x0],
#     [0xffffffff, 0x7fffffff, 0x0, 0x0],
#     [0xffffffff, 0xffffffff, 0x0, 0x0],
#     [0xffffffff, 0xffffffff, 0x1, 0x0],
#     [0xffffffff, 0xffffffff, 0x3, 0x0],
#     [0xffffffff, 0xffffffff, 0x7, 0x0],
#     [0xffffffff, 0xffffffff, 0xf, 0x0],
#     [0xffffffff, 0xffffffff, 0x1f, 0x0],
#     [0xffffffff, 0xffffffff, 0x3f, 0x0],
#     [0xffffffff, 0xffffffff, 0x7f, 0x0],
#     [0xffffffff, 0xffffffff, 0xff, 0x0],
#     [0xffffffff, 0xffffffff, 0x1ff, 0x0],
#     [0xffffffff, 0xffffffff, 0x3ff, 0x0],
#     [0xffffffff, 0xffffffff, 0x7ff, 0x0],
#     [0xffffffff, 0xffffffff, 0xfff, 0x0],
#     [0xffffffff, 0xffffffff, 0x1fff, 0x0],
#     [0xffffffff, 0xffffffff, 0x3fff, 0x0],
#     [0xffffffff, 0xffffffff, 0x7fff, 0x0],
#     [0xffffffff, 0xffffffff, 0xffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x1ffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x3ffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x7ffff, 0x0],
#     [0xffffffff, 0xffffffff, 0xfffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x1fffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x3fffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x7fffff, 0x0],
#     [0xffffffff, 0xffffffff, 0xffffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x1ffffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x3ffffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x7ffffff, 0x0],
#     [0xffffffff, 0xffffffff, 0xfffffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x1fffffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x3fffffff, 0x0],
#     [0xffffffff, 0xffffffff, 0x7fffffff, 0x0],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x0],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x1],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x3],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x7],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0xf],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x1f],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x3f],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x7f],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0xff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x1ff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x3ff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x7ff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0xfff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x1fff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x3fff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x7fff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0xffff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x1ffff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x3ffff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x7ffff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0xfffff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x1fffff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x3fffff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0x7fffff],
#     [0xffffffff, 0xffffffff, 0xffffffff, 0xffffff]
# ]
'''

distributed_mask = [
    [0xff, 0x0, 0x0, 0x0],
    [0xffff, 0x0, 0x0, 0x0],
    [0xffffff, 0x0, 0x0, 0x0],
    [0xffffffff, 0x0, 0x0, 0x0],
    [0xffffffff, 0xff, 0x0, 0x0],
    [0xffffffff, 0xffff, 0x0, 0x0],
    [0xffffffff, 0xffffff, 0x0, 0x0],
    [0xffffffff, 0xffffffff, 0x0, 0x0],
    [0xffffffff, 0xffffffff, 0xff, 0x0],
    [0xffffffff, 0xffffffff, 0xffff, 0x0],
    [0xffffffff, 0xffffffff, 0xffffff, 0x0],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x0],
    [0xffffffff, 0xffffffff, 0xffffffff, 0xff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0xffff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0xffffff]
]

conserved_mask = [
    [0x1, 0x0, 0x0, 0x0],
    [0x11, 0x0, 0x0, 0x0],
    [0x111, 0x0, 0x0, 0x0],
    [0x1111, 0x0, 0x0, 0x0],
    [0x11111, 0x0, 0x0, 0x0],
    [0x111111, 0x0, 0x0, 0x0],
    [0x1111111, 0x0, 0x0, 0x0],
    [0x11111111, 0x0, 0x0, 0x0],
    [0x11111111, 0x1, 0x0, 0x0],
    [0x11111111, 0x11, 0x0, 0x0],
    [0x11111111, 0x111, 0x0, 0x0],
    [0x11111111, 0x1111, 0x0, 0x0],
    [0x11111111, 0x11111, 0x0, 0x0],
    [0x11111111, 0x111111, 0x0, 0x0],
    [0x11111111, 0x1111111, 0x0, 0x0],
    [0x11111111, 0x11111111, 0x0, 0x0],
    [0x11111111, 0x11111111, 0x1, 0x0],
    [0x11111111, 0x11111111, 0x11, 0x0],
    [0x11111111, 0x11111111, 0x111, 0x0],
    [0x11111111, 0x11111111, 0x1111, 0x0],
    [0x11111111, 0x11111111, 0x11111, 0x0],
    [0x11111111, 0x11111111, 0x111111, 0x0],
    [0x11111111, 0x11111111, 0x1111111, 0x0],
    [0x11111111, 0x11111111, 0x11111111, 0x0],
    [0x11111111, 0x11111111, 0x11111111, 0x1],
    [0x11111111, 0x11111111, 0x11111111, 0x11],
    [0x11111111, 0x11111111, 0x11111111, 0x111],
    [0x11111111, 0x11111111, 0x11111111, 0x1111],
    [0x11111111, 0x11111111, 0x11111111, 0x11111],
    [0x11111111, 0x11111111, 0x11111111, 0x111111],
    [0x33333333, 0x13333333, 0x0, 0x0],
    [0x33333333, 0x33333333, 0x0, 0x0],
    [0x33333333, 0x33333333, 0x1, 0x0],
    [0x33333333, 0x33333333, 0x3, 0x0],
    [0x33333333, 0x33333333, 0x13, 0x0],
    [0x33333333, 0x33333333, 0x33, 0x0],
    [0x33333333, 0x33333333, 0x133, 0x0],
    [0x33333333, 0x33333333, 0x333, 0x0],
    [0x33333333, 0x33333333, 0x1333, 0x0],
    [0x33333333, 0x33333333, 0x3333, 0x0],
    [0x33333333, 0x33333333, 0x13333, 0x0],
    [0x33333333, 0x33333333, 0x33333, 0x0],
    [0x33333333, 0x33333333, 0x133333, 0x0],
    [0x33333333, 0x33333333, 0x333333, 0x0],
    [0x33333333, 0x33333333, 0x1333333, 0x0],
    [0x33333333, 0x33333333, 0x3333333, 0x0],
    [0x33333333, 0x33333333, 0x13333333, 0x0],
    [0x33333333, 0x33333333, 0x33333333, 0x0],
    [0x33333333, 0x33333333, 0x33333333, 0x1],
    [0x33333333, 0x33333333, 0x33333333, 0x3],
    [0x33333333, 0x33333333, 0x33333333, 0x13],
    [0x33333333, 0x33333333, 0x33333333, 0x33],
    [0x33333333, 0x33333333, 0x33333333, 0x133],
    [0x33333333, 0x33333333, 0x33333333, 0x333],
    [0x33333333, 0x33333333, 0x33333333, 0x1333],
    [0x33333333, 0x33333333, 0x33333333, 0x3333],
    [0x33333333, 0x33333333, 0x33333333, 0x13333],
    [0x33333333, 0x33333333, 0x33333333, 0x33333],
    [0x33333333, 0x33333333, 0x33333333, 0x133333],
    [0x33333333, 0x33333333, 0x33333333, 0x333333],
    [0x77777777, 0x77777777, 0x17777, 0x0],
    [0x77777777, 0x77777777, 0x37777, 0x0],
    [0x77777777, 0x77777777, 0x77777, 0x0],
    [0x77777777, 0x77777777, 0x177777, 0x0],
    [0x77777777, 0x77777777, 0x377777, 0x0],
    [0x77777777, 0x77777777, 0x777777, 0x0],
    [0x77777777, 0x77777777, 0x1777777, 0x0],
    [0x77777777, 0x77777777, 0x3777777, 0x0],
    [0x77777777, 0x77777777, 0x7777777, 0x0],
    [0x77777777, 0x77777777, 0x17777777, 0x0],
    [0x77777777, 0x77777777, 0x37777777, 0x0],
    [0x77777777, 0x77777777, 0x77777777, 0x0],
    [0x77777777, 0x77777777, 0x77777777, 0x1],
    [0x77777777, 0x77777777, 0x77777777, 0x3],
    [0x77777777, 0x77777777, 0x77777777, 0x7],
    [0x77777777, 0x77777777, 0x77777777, 0x17],
    [0x77777777, 0x77777777, 0x77777777, 0x37],
    [0x77777777, 0x77777777, 0x77777777, 0x77],
    [0x77777777, 0x77777777, 0x77777777, 0x177],
    [0x77777777, 0x77777777, 0x77777777, 0x377],
    [0x77777777, 0x77777777, 0x77777777, 0x777],
    [0x77777777, 0x77777777, 0x77777777, 0x1777],
    [0x77777777, 0x77777777, 0x77777777, 0x3777],
    [0x77777777, 0x77777777, 0x77777777, 0x7777],
    [0x77777777, 0x77777777, 0x77777777, 0x17777],
    [0x77777777, 0x77777777, 0x77777777, 0x37777],
    [0x77777777, 0x77777777, 0x77777777, 0x77777],
    [0x77777777, 0x77777777, 0x77777777, 0x177777],
    [0x77777777, 0x77777777, 0x77777777, 0x377777],
    [0x77777777, 0x77777777, 0x77777777, 0x777777],
    [0xffffffff, 0xffffffff, 0x7ffffff, 0x0],
    [0xffffffff, 0xffffffff, 0xfffffff, 0x0],
    [0xffffffff, 0xffffffff, 0x1fffffff, 0x0],
    [0xffffffff, 0xffffffff, 0x3fffffff, 0x0],
    [0xffffffff, 0xffffffff, 0x7fffffff, 0x0],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x0],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x1],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x3],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x7],
    [0xffffffff, 0xffffffff, 0xffffffff, 0xf],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x1f],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x3f],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x7f],
    [0xffffffff, 0xffffffff, 0xffffffff, 0xff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x1ff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x3ff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x7ff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0xfff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x1fff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x3fff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x7fff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0xffff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x1ffff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x3ffff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x7ffff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0xfffff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x1fffff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x3fffff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0x7fffff],
    [0xffffffff, 0xffffffff, 0xffffffff, 0xffffff]
]

# exec: ["test_one_kernel.out", "test_one_model.out"]
# Memory access fault on test_one_kernel.out of MobileNet
def test_cumask(exec):
    os.environ["ENABLE_CUMASK"] = "1"
    name = ["test_vgg", "test_darknet", "test_mobilenet", "test_resnet152", "test_resnet50", "test_bert", "test_alexnet"]
    for i in range(7):
        cmd = "./" + exec + " " + str(i)
        os.system("echo @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        os.system("echo " + name[i])
        os.system("echo @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # os.system("echo packed cu mask")
        # os.system("echo")
        # for m in packed_mask:
        #     os.environ["CUMASK_0"] = hex(m[0])
        #     os.environ["CUMASK_1"] = hex(m[1])
        #     os.environ["CUMASK_2"] = hex(m[2])
        #     os.environ["CUMASK_3"] = hex(m[3])
        #     os.system(cmd)
        
        os.system("echo ========================================================================================")
        os.system("echo distributed cu mask")
        os.system("echo")
        for m in distributed_mask:
            os.environ["CUMASK_0"] = hex(m[0])
            os.environ["CUMASK_1"] = hex(m[1])
            os.environ["CUMASK_2"] = hex(m[2])
            os.environ["CUMASK_3"] = hex(m[3])
            os.system(cmd)

        # os.system("echo ========================================================================================")
        # os.system("echo conserved cu mask")
        # os.system("echo")
        # for m in conserved_mask:
        #     os.environ["CUMASK_0"] = hex(m[0])
        #     os.environ["CUMASK_1"] = hex(m[1])
        #     os.environ["CUMASK_2"] = hex(m[2])
        #     os.environ["CUMASK_3"] = hex(m[3])
        #     os.system(cmd)

def test_latency():
    name = ["test_vgg", "test_darknet", "test_mobilenet", "test_resnet152", "test_resnet50", "test_alexnet", "test_bert"]
    right_size = [112, 112, 112, 64, 64, 104]
    os.environ["ENABLE_CUMASK"] = "1"
    for i in range(6):
        cmd = "./test_one_model.out " + str(i)
        index = int(right_size[i] / 8 - 1)
        os.environ["CUMASK_0"] = hex(distributed_mask[index][0])
        os.environ["CUMASK_1"] = hex(distributed_mask[index][1])
        os.environ["CUMASK_2"] = hex(distributed_mask[index][2])
        os.environ["CUMASK_3"] = hex(distributed_mask[index][3])
        os.system("echo " + name[i])
        for cnt in range(10):
            os.system(cmd)
        os.system("echo")

def test_right_size():
    right_size = [64, 72, 80, 88, 96, 104, 112]
    for i in range(6):
        for j in range(6):
            for size in right_size:
                cmd = "./test_right_size.out " + str(i) + " " + str(j) + " " + str(size) + " >> tmp.log"
                for cnt in range(20):
                    os.system(cmd)
                with open("tmp.log") as f:
                    sum = 0.0
                    for line in f:
                        sum += float(line)
                    sum = sum / 20
                    os.system("echo " + str(sum))
                os.system("rm tmp.log")
        os.system("echo")

def compare_service():
    for i in [1, 2]:
    # for i in range(6):
        for j in range(5):
            cmd = "./compare_service.out " + str(i) + " " + str(j) + " >> compare_service.log"
            os.system(cmd)
            time.sleep(120)

def compare_service_cumask():
    os.environ["CUMASK0_0"] = "0x33333333"
    os.environ["CUMASK0_1"] = "0x33333333"
    os.environ["CUMASK0_2"] = "0x33333333"
    os.environ["CUMASK0_3"] = "0x00333333"

    os.environ["CUMASK1_0"] = "0xcccccccc"
    os.environ["CUMASK1_1"] = "0xcccccccc"
    os.environ["CUMASK1_2"] = "0xcccccccc"
    os.environ["CUMASK1_3"] = "0x00cccccc"
    for i in range(6):
        os.environ["ENABLE_CUMASK"] = "1"

        cmd = "./compare_service.out " + str(i) + " 0 >> compare_service.log"
        os.system(cmd)
        time.sleep(120)

        del os.environ["ENABLE_CUMASK"]
        for j in range(5):
            cmd = "./compare_service.out " + str(i) + " " + str(j) + " >> compare_service.log"
            os.system(cmd)
            time.sleep(120)

def test_co_kernels(enbale_cuMask):
    if (enbale_cuMask):
        os.environ["ENABLE_CUMASK"] = "1"
        os.environ["CUMASK0_0"] = "0x33333333"
        os.environ["CUMASK0_1"] = "0x33333333"
        os.environ["CUMASK0_2"] = "0x33333333"
        os.environ["CUMASK0_3"] = "0x00333333"

        os.environ["CUMASK1_0"] = "0xcccccccc"
        os.environ["CUMASK1_1"] = "0xcccccccc"
        os.environ["CUMASK1_2"] = "0xcccccccc"
        os.environ["CUMASK1_3"] = "0x00cccccc"
    for i in range(2, 10):
        for j in range(8):
            cmd = "./test_co_kernels.out " + str(i) + " " + str(j) + " >> test_co_kernels.log"
            os.system(cmd)
        time.sleep(120)

if __name__ == '__main__':
    # test_cumask("test_one_model.out")
    # test_latency()
    # compare_service()
    compare_service_cumask()
    # test_co_kernels(0)
    # test_co_kernels(1)
    # test_right_size()
    # for i in range(6):
    # i = 2
    # for j in range(6):
    #     # for cnt in range(20):
    #     os.system("./test_priority_service.out " + str(i) + " " + str(j))
    #     os.system("echo")