from netflix import TopNRecommended


print("ورود ایدی کاربر جهت پیش نهاد فیلم")



while True:
    num = int(input())
    if num == 0:
        break
    print(TopNRecommended(num))


