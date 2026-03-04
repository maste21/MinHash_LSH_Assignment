import numpy as np
import random
from itertools import combinations

def read_doc(f):
    return open(f).read().lower()

def char_kgrams(t,k):
    s=set()
    for i in range(len(t)-k+1):
        s.add(t[i:i+k])
    return s

def word_kgrams(t,k):
    w=t.split()
    s=set()
    for i in range(len(w)-k+1):
        s.add(" ".join(w[i:i+k]))
    return s

def jaccard(a,b):
    return len(a&b)/len(a|b)

def build_matrix(docs):
    allg=set()
    for d in docs:
        allg|=d
    allg=list(allg)
    M=np.zeros((len(allg),len(docs)))
    for i,g in enumerate(allg):
        for j,d in enumerate(docs):
            if g in d:
                M[i][j]=1
    return M

def minhash(M,t):
    m=M.shape[0]
    sig=np.ones((t,M.shape[1]))*999999
    a=[random.randint(1,10000) for _ in range(t)]
    b=[random.randint(1,10000) for _ in range(t)]

    for r in range(m):
        for i in range(t):
            h=(a[i]*r+b[i])%10007
            for c in range(M.shape[1]):
                if M[r][c]==1 and h<sig[i][c]:
                    sig[i][c]=h
    return sig

def approx_j(sig,i,j):
    t=sig.shape[0]
    c=0
    for k in range(t):
        if sig[k][i]==sig[k][j]:
            c+=1
    return c/t

def lsh(sig,b,r):
    n=sig.shape[1]
    cand=set()
    for band in range(b):
        bucket={}
        s=band*r
        e=s+r

        for col in range(n):
            key=tuple(sig[s:e,col])
            bucket.setdefault(key,[]).append(col)

        for v in bucket.values():
            if len(v)>1:
                for p in combinations(v,2):
                    cand.add(p)

    return cand


print("\n==============================")
print("PART 1: K-GRAM JACCARD")
print("==============================\n")

D1=read_doc("D1.txt")
D2=read_doc("D2.txt")
D3=read_doc("D3.txt")
D4=read_doc("D4.txt")

docs=[D1,D2,D3,D4]

c2=[char_kgrams(d,2) for d in docs]
c3=[char_kgrams(d,3) for d in docs]
w2=[word_kgrams(d,2) for d in docs]

print("Character 2-grams\n")
for i,j in combinations(range(4),2):
    print(f"D{i+1} vs D{j+1} = {jaccard(c2[i],c2[j])}")

print("\nCharacter 3-grams\n")
for i,j in combinations(range(4),2):
    print(f"D{i+1} vs D{j+1} = {jaccard(c3[i],c3[j])}")

print("\nWord 2-grams\n")
for i,j in combinations(range(4),2):
    print(f"D{i+1} vs D{j+1} = {jaccard(w2[i],w2[j])}")


print("\n==============================")
print("PART 2: MINHASH")
print("==============================\n")

M=build_matrix(c3)

for t in [20,60,150,300,600]:
    sig=minhash(M,t)
    print(f"t = {t} -> {approx_j(sig,0,1)}")


print("\n==============================")
print("PART 3: LSH")
print("==============================\n")

sig=minhash(M,160)
cand=lsh(sig,20,8)

for p in cand:
    print(f"Candidate pair: D{p[0]+1} , D{p[1]+1}")


print("\nLSH Probability Values\n")

r=8
b=20

sims=[0.977979274611399,0.5803571428571429,0.3050847457627119,
      0.5680473372781065,0.30590339892665475,0.31212381771281167]

pairs=["D1-D2","D1-D3","D1-D4","D2-D3","D2-D4","D3-D4"]

for p,s in zip(pairs,sims):
    prob=1-(1-(s**b))**r
    print(p,"Probability =",prob)


print("\n==============================")
print("PART 4: MOVIELENS MINHASH")
print("==============================\n")

users={}

for line in open("u.data"):
    u,m,r,t=line.split()
    u=int(u)
    m=int(m)
    users.setdefault(u,set()).add(m)

true_pairs=set()

for a,b in combinations(users.keys(),2):
    if jaccard(users[a],users[b])>=0.5:
        true_pairs.add((a,b))

print("Exact pairs >=0.5 =",len(true_pairs))


def user_minhash(users,t):

    movies=set()
    for v in users.values():
        movies|=v

    movies=list(movies)
    uid=list(users.keys())

    M=np.zeros((len(movies),len(uid)))

    for i,m in enumerate(movies):
        for j,u in enumerate(uid):
            if m in users[u]:
                M[i][j]=1

    sig=minhash(M,t)

    return sig,uid


def estimated_pairs(sig,uid,thr):

    est=set()

    for i,j in combinations(range(len(uid)),2):
        if approx_j(sig,i,j)>=thr:
            est.add((uid[i],uid[j]))

    return est


for t in [50,100,200]:

    print(f"\nExperiment t = {t}")

    fp_list=[]
    fn_list=[]

    for run in range(5):

        sig,uid=user_minhash(users,t)
        est=estimated_pairs(sig,uid,0.5)

        fp=len(est-true_pairs)
        fn=len(true_pairs-est)

        fp_list.append(fp)
        fn_list.append(fn)

        print(f"Run {run+1}: FP={fp} FN={fn}")

    print("Average FP =",sum(fp_list)/5)
    print("Average FN =",sum(fn_list)/5)


print("\n==============================")
print("PART 5: LSH MOVIELENS")
print("==============================\n")

configs=[(50,5,10),(100,5,20),(200,5,40),(200,10,20)]

for t,r,b in configs:

    print(f"\nConfiguration: t={t} r={r} b={b}")

    fp_list=[]
    fn_list=[]

    for run in range(5):

        sig,uid=user_minhash(users,t)
        cand=lsh(sig,b,r)

        fp=0
        fn=0

        for i,j in cand:
            if jaccard(users[uid[i]],users[uid[j]])<0.6:
                fp+=1

        for a,b2 in true_pairs:
            if (a,b2) not in cand and (b2,a) not in cand:
                if jaccard(users[a],users[b2])>=0.6:
                    fn+=1

        fp_list.append(fp)
        fn_list.append(fn)

        print(f"Run {run+1}: FP={fp} FN={fn}")

    print("Average FP =",sum(fp_list)/5)
    print("Average FN =",sum(fn_list)/5)