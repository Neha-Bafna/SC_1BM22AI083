#Merge sort
def merge(a,l,m,r):
    i=l
    j=m+1
    b=[]
    while i<=m and j<=r:
        if a[i]<=a[j]:
            b.append(a[i])
            i=i+1
        else:
            b.append(a[j])
            j=j+1
    if i>m:
        while j<=r:
            b.append(a[j])
            j=j+1
    if j>r:
        while i<=m:
            b.append(a[i])
            i=i+1
    pos=0
    for k in range(l,r+1):
        a[k]=b[pos]
        pos+=1
def mergesort(a,l,r):
    m=(l+r)//2
    if l<r:
        mergesort(a,l,m)
        mergesort(a,m+1,r)
        merge(a,l,m,r)
A=[1, 5, 3, 2, 6, 4]
mergesort(A,0,len(A)-1)
print(A)
