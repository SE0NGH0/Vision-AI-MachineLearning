# numpy와 matplotlib으로 임의의 이미지 데이터를 생성하고, 서브플롯에서 다양한 매개변수로 이미지를 다르게 렌더링함.

import numpy as np
import matplotlib.pyplot as plt

# 임의의 5x5 이미지 데이터 생성
# data = np.random.random((5, 5))
data = np.array([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20],
                 [21, 22, 23, 24, 25]])

# 그래프 간 간격 설정
plt.subplots_adjust(wspace=0.4, hspace=0.6)

# X: 이미지 데이터
plt.subplot(2, 4, 1)
plt.imshow(data)
plt.title('X: Image Data', fontsize=8)

# cmap (컬러 맵)-데이터를 색상으로 표현하는 방법을 정의: 'viridis'
plt.subplot(2, 4, 2)
plt.imshow(data, cmap='viridis')
plt.title('cmap: Viridis', fontsize=8)

# interpolation (보간법)-이미지를 확대/축소할 때 픽셀 간의 값들을 계산하는 방법: 'bicubic' - 4x4 픽셀을 이용해 부드럽게 보간(부드러운 이미지)
plt.subplot(2, 4, 3)
plt.imshow(data, interpolation='bicubic')
plt.title('Interpolation: Bicubic', fontsize=8)

# aspect (가로세로 비율): 'equal'-데이터의 실제 비율을 유지 / 'auto': 그래프의 크기에 맞춰 비율을 조정
plt.subplot(2, 4, 4)
plt.imshow(data, aspect='equal')
plt.title('Aspect: Equal', fontsize=8)

# extent: 좌표 범위 지정 - 사용자 정의 좌표로 변경
plt.subplot(2, 4, 5)
plt.imshow(data, extent=[-2, 2, -2, 2])
plt.title('Extent: [-2, 2, -2, 2]', fontsize=8)

# vmin, vmax: 컬러 맵 범위 지정 - 데이터를 컬러 맵에 매핑할 때 특정 범위를 설정/ 데이터 강조 범위를 조정
plt.subplot(2, 4, 6)
plt.imshow(data, vmin=0.2, vmax=0.8)
plt.title('vmin=0.2, vmax=0.8', fontsize=8)

# origin: 이미지 원점 위치 - 'upper': (0, 0)이 이미지의 왼쪽 위/'lower': (0, 0)이 이미지의 왼쪽 아래
plt.subplot(2, 4, 7)
plt.imshow(data, origin='lower')
plt.title('Origin: Lower', fontsize=8)

# 그 외 다양한 매개변수 활용
plt.subplot(2, 4, 8)
plt.imshow(data, alpha=0.5, cmap='hot', interpolation='nearest')
plt.title('Other Parameters', fontsize=8)

plt.show()