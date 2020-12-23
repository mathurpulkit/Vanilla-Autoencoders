import fc
import cnn
print("Welcome to Autoencoders! Please select your choice: ")
print("a. FC based")
print("b. CNN based")
ans = input("Enter your choice: ")
if ans == 'a' or ans == 'A':
    fc.main()
elif ans == 'b' or ans == 'B':
    cnn.main()
