#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)


# Obtaining the  3 x 2 grid
fig = plt.figure()

axis_1 = fig.add_subplot(3, 2, 1)
axis_1.plot(y0, c='r')
axis_1.set_xlim(left=0, right=10)
axis_1.set_ylim(bottom=0, top=1000)

axis_2 = fig.add_subplot(3, 2, 2)
axis_2.scatter(x1, y1, c='m')
axis_2.set_xlabel('Height (in)', fontsize='x-small')
axis_2.set_ylabel('Weight (lbs)', fontsize='x-small')
axis_2.set_title("Men's Height vs Weight", fontsize='x-small')

axis_3 = fig.add_subplot(3, 2, 3)
axis_3.plot(x2, y2)
axis_3.set_xlabel('Time (years)', fontsize='x-small')
axis_3.set_ylabel('Fraction Remaining', fontsize='x-small')
axis_3.set_title("Exponential Decay of C-14", fontsize='x-small')
axis_3.set_yscale("log")
axis_3.set_xlim(left=0, right=28650)

axis_4 = fig.add_subplot(3, 2, 4)
axis_4.plot(x3, y31, c='r', linestyle='--', label='C-14')
axis_4.plot(x3, y32, c='g', label='Ra-226')
axis_4.set_xlabel('Time (years)', fontsize='x-small')
axis_4.set_ylabel('Fraction Remaining', fontsize='x-small')
axis_4.set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
axis_4.legend()
axis_4.set_xlim(left=0, right=20000)
axis_4.set_ylim(bottom=0, top=1)

axis_5 = fig.add_subplot(3, 1, 3)
axis_5.hist(student_grades,
         bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
         edgecolor='black')
axis_5.set_xlabel('Grades', fontsize='x-small')
axis_5.set_ylabel('Number of Students', fontsize='x-small')
axis_5.set_title("Project A", fontsize='x-small')
axis_5.set_xlim((0, 100))
axis_5.set_ylim((0, 30))

fig.suptitle("All in One")
fig.subplots_adjust(top=0.95)
plt.tight_layout()
plt.show();
