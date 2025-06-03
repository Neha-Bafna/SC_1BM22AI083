import concurrent.futures

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def parallel_merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    with concurrent.futures.ThreadPoolExecutor() as executor:
        left_future = executor.submit(parallel_merge_sort, arr[:mid])
        right_future = executor.submit(parallel_merge_sort, arr[mid:])
        left = left_future.result()
        right = right_future.result()

    return merge(left, right)

# Example usage:
arr = [38, 27, 43, 3, 9, 82, 10]
sorted_arr = parallel_merge_sort(arr)
print("Sorted array:", sorted_arr)
