import random
import time

print('로또번호를 추첨합니다')

for i in range(0,6):
    num = random.randint(1,45)
    print(num, end=' ')

time.sleep(1)
print('보너스번호: %2d'%random.randint(1,45))
