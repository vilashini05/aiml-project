print("Hello Python")

mark1=int(input("Enter sub1 :"))
mark2=int(input("Enter sub2 :"))
mark3=int(input("Enter sub3 :"))
mark4=int(input("Enter sub4 :"))
mark5=int(input("Enter sub5 :"))

total=mark1+mark2+mark3+mark4+mark5
avg=total/5

if total>=450:
    print("O Grade")
elif total>=400:
    print("A+ Grade")
elif total>=350:
    print("A Grade")
elif total>=300:
    print("B+ Grade")
elif total >=250:
    print("B Grade")
else:
    print("Failed")


print("Total marks obtained out of 500 :",total)
print("Average marks :",avg)
