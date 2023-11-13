# %%
# import all common packages
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import math
import os
import sys

# %%
def sum_odd_digits(number):
    #TODO: implement this function
    strs = str(number)
    sum = 0
    for i in strs:
        if int(i) % 2 != 0:
            sum += int(i)
    return sum

def sum_even_digits(number):
    #TODO: implement this function
    strs = str(number)
    sum = 0
    for i in strs:
        if int(i) % 2 == 0:
            sum += int(i)
    return sum

def sum_all_digits(number):
    #TODO: implement this function
    strs = str(number)
    sum = 0 
    for i in strs:
        sum += int(i)
    return sum

# %%
# Consider the series

# total = 1/1 + 1/2 + 1/3 + 1/4 + 1/5 .... + 1/N
# What is the maximum number of terms added (i.e. the value of N) such that total < 5.0?

def check_series():
    total = 0
    n = 0
    while total < 5.0:
        n += 1
        total += 1 / n
    return n - 1

check_series()

# %%
# Write a function called is_increasing that takes a sequence (of numbers) and returns True iff the elements in the array are in (non-strict) increasing order. This means that every element is less than or equal to the next one after it. For example,

# for [1, 5, 9] the function should return True
# for [3, 3, 4] the function should return True
# for [3, 4, 2] the function should return False

def is_increasing(sequence):
    #TODO: implement this function
    return all(sequence[i] <= sequence[i+1] for i in range(len(sequence)-1))

# %%
# Write a function most_average(numbers) which finds and returns the number in the input that is closest to the average of the numbers. (You can assume that the argument is a sequence of numbers.) By closest, we mean the one that has the smallest absolute difference from the average. You can use the built-in function abs to find the absolute value of a difference. For example, most_average([1, 2, 3, 4, 5]) should return 3 (the average of the numbers in the list is 3.0, and 3 is clearly closest to this). most_average([3, 4, 3, 1]) should also return 3 (the average is 2.75, and 3 is closer to 2.75 than is any other number in the list).


def most_average(numbers):
    #TODO: implement this function
    avg = sum(numbers)/len(numbers)
    diff = [abs(x-avg) for x in numbers]
    return numbers[diff.index(min(diff))]

# %%
# Write two functions, smallest_greater(seq, value) and greatest_smaller(seq, value), that take as argument a sequence and a value, and find the smallest element in the sequence that is greater than the given value, and the greatest element in the sequence that is smaller than the given value, respectively.

# For example, if the sequence is [13, -3, 22, 14, 2, 18, 17, 6, 9] and the target value is 4, then the smallest greater element is 6 and the greatest smaller element is 2.

def smallest_greater(seq, value):
    #TODO: implement this function
    seqs = sorted(seq)
    for i in seqs:
        if i > value:
            return i


def greatest_smaller(seq, value):
    #TODO: implement this function
    seqs = sorted(seq, reverse=True)
    for i in seqs:
        if i < value:
            return i

# %%
# If the same value appears more than once in a sequence, we say that all copies of it except the first are duplicates. For example, in [-1, 2, 4, 2, 0, 4], the second 2 and second 4 are duplicates; in the string "Immaterium", the 'm' is duplicated twice (but the 'i' is not a duplicate, because 'I' and 'i' are different characters).

def count_duplicates(seq):
    #TODO: implement this function
    count = 0
    for i in range(len(seq)):
        for j in range(i+1, len(seq)):
            if seq[i] == seq[j]:
                count += 1
                break 
    return count

# %%
def count_in_bin(values, lower, upper):
    #TODO: implement this function
    count = 0
    for value in values:
        if lower < value <= upper:
            count += 1
    return count


# Write a function histogram(values, dividers) that takes as argument a sequence of values and a sequence of bin dividers, and returns the histogram as a sequence of a suitable type (say, an array) with the counts in each bin. The number of bins is the number of dividers + 1; the first bin has no lower limit and the last bin has no upper limit. As in (a), elements that are equal to one of the dividers are counted in the bin below.
def histogram(values, dividers):
    #TODO: implement this function
    counts = []
    for i in range(len(dividers) + 1):
        if i == 0:
            counts.append(count_in_bin(values, float('-inf'), dividers[i]))
        elif i == len(dividers):
            counts.append(count_in_bin(values, dividers[i-1], float('inf')))
        else:
            counts.append(count_in_bin(values, dividers[i-1], dividers[i]))
    return counts

# %%
# In the unicode encoding system (which is used by python 3),  the string "Dog" is represented by the following sequence of numbers (character codes):

# 68,111,103


string = 'Dog'
print([ord(c) for c in string])

# ord to string
print(''.join([chr(c) for c in [86,101,114,121,32,72,97,114,100,32,69,120,97,109]]))

# %%
s = "problem"
s[1] + s[5] + s[6]

# %%
def f(s):
    for elem1 in s:
        for elem2 in s[::-1]:
            if elem1 == elem2:
                return elem1
f("abcbdc")

# %%

'a way' < 'away'

# %%
len("1.5" + "1.5") == 3

# %%
def count_capitals(string):
    #TODO: implement this function
    cnt = 0
    for i in string:
        if i.isupper():
            cnt += 1
    return cnt

def count(seq, prop):
    cnt = 0
    for i in seq:
        if prop(i):
            cnt += 1
    return cnt

# %%
def sum_odd_digits(number):
    dsum = 0 # digit sum
    strs = str(number)
    for i in strs:
        if int(i) % 2 != 0:
            dsum += int(i)
    return dsum

def sum_even_digits(number):
    dsum = 0 # digit sum
    strs = str(number)
    for i in strs:
        if int(i) % 2 == 0:
            dsum += int(i)
    return dsum

print(sum_odd_digits(12345))
print(sum_odd_digits(456789))
print(sum_even_digits(12345))
print(sum_even_digits(456789))


# %%
def count_kmer(sequence, k):
    """ counting occurence
    of all distinct kmers"""
    distinct_kmers = {}
    result = []
    for index in range(len(sequence)-k+1):
        kmer = sequence[index:index+k]
        if kmer not in distinct_kmers:
            distinct_kmers[kmer] = 1
        else:
            distinct_kmers[kmer] += 1
    for key, value in distinct_kmers.items():
        result.append((key, value))
    return result

print (sorted(count_kmer("AGAGACCCCCT", 3)))
print (sorted(count_kmer("AGAGACCCCCT", 2)))
print (sorted(count_kmer("A", 1)))
print (sorted(count_kmer("A", 2)))

# %%
## a)
def caesar_shift(string, shift):
    #TODO: implement this function
    result = ""
    for s in string:
        if s.isalpha():
            if s.isupper():
                result += chr((ord(s) + shift - 65) % 26 + 65)
            else:
                result += chr((ord(s) + shift - 97) % 26 + 97)
        else:
            result += s
    return result

## b)
def decrypt_5(code):
    """Prints out the first 5 words decrypted using successively larger shifts
    """
    #TODO: implement this function
    # from -5 to -1
    for i in range(-5, 0):
        print(caesar_shift(code, i))
    pass

def decrypt_search(code):
    """Decrypt the message using increasing shift values whilst searching for 40 
    common three letter words. Return the shift value that gives the highest number 
    of different three letter words.
    """
    #TODO: implement this function
    # common words = [the,and,for,are,but,not,you,all,any,can, her,was,one,our,out,day,get,has,him,his, how,man,new,now,old,see,two,way,who,boy, did,its,let,put,say,she,too,use,dad,mom]
    common_word_list = ["the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use", "dad", "mom"]
    # Return the shift value that gives the highest number of different three letter words.
    countings = []
    for i in range(-26,26):
        result = caesar_shift(code, i)
        words = result.split()
        count_sum = {}
        for word in words:
            word = word.lower()
            if word in common_word_list:
                if word not in count_sum:
                    count_sum[word] = 1
                else:
                    count_sum[word] += 1
        # sord by value descending
        count_sum = sorted(count_sum.items(), key=lambda x: x[1], reverse=True)
        # sum up top 3
        top3 = 0
        for key, value in count_sum[:3]:
            top3 += value
        countings.append(top3)
        
    
    return 26 - countings.index(max(countings))
                   
    

    
def decrypt_find_e(code):
    """Decrypt Caesar ciphered message by finding most frequently occuring letter
    assume it is an "e" and return the corresponding shift.
    """
    #TODO: implement this function
    countings = []
    for i in range(-26,26):
        result = caesar_shift(code, i)
        count_e = 0
        for s in result:
            if s == "e" or s == "E":
                count_e += 1
        countings.append(count_e)
        
    
    return 26 - countings.index(max(countings))


## a)
print(caesar_shift("Et tu, Brutus!", 3))
print(caesar_shift("IBM", -1))
print(caesar_shift("COMP1730 is great!", 25))
print(caesar_shift("COMP1730 is great!", -25))
print(caesar_shift("uwu", -27))
print(caesar_shift("You Could Use Facts To Prove Anything That's Even Remotely True.", 29))

## b)
message1 = '''Awnhu pk neoa wjz awnhu pk xaz
              Iwgao w iwj dawhpdu, xqp okyewhhu zawz'''
message2 = '"Jcstghipcsxcv xh iwpi etctigpixcv fjpaxin du zcdlatsvt iwpi vgdlh ugdb iwtdgn, egprixrt, rdckxrixdc, phhtgixdc, tggdg pcs wjbxaxipixdc." (Gjat 7: Jht p rdadc puitg pc xcstetcstci rapjht id xcigdsjrt p axhi du epgixrjapgh, pc peedhxixkt, pc pbeaxuxrpixdc dg pc xaajhigpixkt fjdipixdc. Ugdb Higjcz & Lwxit, "Iwt Tatbtcih du Hinat".)'
message3 = "Cywo cmsoxdscdc gybu cy rkbn drobo sc xy dswo vopd pyb cobsyec drsxusxq. (kddbsledon dy Pbkxmsc Mbsmu)"
print(decrypt_search(message1))
print(decrypt_search(message2))
print(decrypt_search(message3))
print(decrypt_find_e(message1))
print(decrypt_find_e(message2))

# %%
# If a word begins with a vowel, append "yay" to the end of the word.

# If a word begins with a consonant, remove all the consonants from the beginning up to the first vowel and append them to the end of the word. Finally, append "ay" to the end of the word.

def to_pig_latin(string):
    #TODO: implement this function
    # if is not a word, return itself

    if not string.isalpha():
        return string
    
    result = ""
    
     # the initial consonant sound is transposed to the end of the word
    if string[0] not in "aeiou":
        for i in range(len(string)):
            if string[i] in "aeiou":
                result = string[i:] + string[:i] + "ay"
                break
    else:
        result = string + "yay"
    return result

print(to_pig_latin('dog'))
print(to_pig_latin('scratch'))
print(to_pig_latin('is'))
print(to_pig_latin('apple'))
print(to_pig_latin('1287643'))

# %%
# Given a function p(N) <= 5/(6*N), q(N) <= 1/4. N is from {4,8,12,16,20,24,28,32,36,40}. 
# Plot the function p(N) and q(N) in the same figure. Use different colors for the two functions and add a legend to the figure.

import matplotlib.pyplot as plt
import numpy as np

N = np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40])

p = 5 / (6 * N)
q = 0.25 * np.ones(len(N))

plt.plot(N, p, label='p(N)')
plt.plot(N, q, label='q(N)')

plt.xlabel('N')
plt.ylabel('Function values')
plt.title('Plot of p(N) and q(N)')
plt.legend()

plt.show()

# %%
max(1,6)

# %%
list1=[1,2,3,4,5]
list2=list1
list3=list2
list1.extend([6,7,8])
list2.reverse()
list3.remove(6)

list1

# %%
lsit = [(n*n) for n in range(1234) if (n*n)%2==0]
len(lsit) 

# %%
def perfect_shuffle_in_place(a_list):
    #TODO: implement this function
    perfect_list = a_list.copy()
    # split into two lists
    list1 = perfect_list[0:len(perfect_list)//2]
    list2 = perfect_list[len(perfect_list)//2:]
    # shuffle
    perfect_list = []
    for i in range(len(list1)):
        perfect_list.append(list1[i])
        perfect_list.append(list2[i])
    a_list[:] = perfect_list


def count_shuffles(a_list):
    #TODO: imeplement this function
    count = 0
    perfect_list = a_list.copy()
    while True:
        count += 1
        perfect_shuffle_in_place(perfect_list)
        if perfect_list == a_list:
            break
    return count

# %%
def nesting_depth(a_list):
    #TODO: imeplement this function
    max_depth = 0
    for e in a_list:
        if type(e) == list:
            depth = nesting_depth(e)
            if depth > max_depth:
                max_depth = depth
    return max_depth 

# %%
a_dict = { 'a' : ['a'],  'b' : ['b'] }
a_dict['a'] = a_dict.copy()
a_dict['c'] = a_dict
a_dict['a']['b'].append('c')
a_dict['c']['a']['b'] = []


a_dict['c']['b']

# %%
import numpy as np

# 三个列向量，假设为 a, b, c
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

# 构建 3x3 矩阵
matrix = np.column_stack((a, b, c))

# 使用叉乘和点乘计算行列式
determinant = np.dot(a, np.cross(b, c))
print("行列式的值为:", determinant)

# %%
def invert_dictionary(d):
    d_inv = {}
    for key, value in d.items():
        if value not in d_inv:
            d_inv[value] = key
        else:
            d_inv[value] = key
    return d_inv

# %%
def invert_dictionary(d):
    d_inv = {}
    for key, value in d.items():
        if value not in d_inv:
            d_inv[value] = key
        else:
            d_inv[value] = key
    return d_inv

# %%
def closed_sets(permutation):
    #TODO: implement this function
    checked = []
    result = []
    for key in permutation.keys():
        if key not in checked:
            current_res = [key]
            current_search = key
            while current_search in permutation.keys() and current_search not in current_res:
                checked.append(current_search)
                nextkey = permutation[current_search]
                current_res.append(nextkey)
                current_search = nextkey
            result.append(current_res)
    return result

p1 = { 'alice' : 'carol', 'bob' : 'bob', 'carol' : 'eve',
       'dave' : 'dave', 'eve' : 'alice' }
closed_sets(p1)

# %%
def count_repetitions(string, substring):
    '''
    Count the number of repetitions of substring in string. Both
    arguments must be strings.
    '''
    return string.count(substring)

# %%
print(count_repetitions("aabsabs","abs"))

# %%
def remove_substring_everywhere(string, substring):
    '''
    Remove all occurrences of substring from string, and return
    the resulting string. Both arguments must be strings.
    '''
    return string.replace(substring, "")

# %%
print(remove_substring_everywhere("aabsabs","abs"))

# %%

def funA(alist):
    if len(alist) == 0:
        return alist
    else:
        return funA(alist[1:]) + [alist[0]]

def funB(alist):
    if len(alist) > 0:
        x = alist.pop(-1)
        alist = funB(alist)
        alist.insert(0,x)
    return alist

a = [1,2,3]
b = funB(a)
c = funA(a)
print(a,b,c)

# %%
def f(x):
    print(y)
    if y < 1:
        z = 1
        return x ** z
    else:
        return x ** y

y = 2
print(f(2))

# %%
def f(x):
    global y
    print(y)
    if y < 1:
        y = 1
    return x ** y

y = 2
print(f(2))