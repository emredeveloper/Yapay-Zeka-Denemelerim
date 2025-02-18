def slice_windows(windows_size, data):
    if windows_size > len(data):
        raise ValueError("Window size cannot be larger than the data size")
    
    
    
    avarage = []
    for i in range(len(data) - windows_size + 1):
        window = data[i:i+windows_size]
        avarage.append(sum(window) / windows_size)
        
    return avarage

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
windows_size = 2

result = slice_windows(windows_size, data)
print(result)