def drop_useless(data, array):
    data = data.drop(array, axis=1) # College ID useless?
    return data

def date_update(data):
    col = data["DOB"]
    i = 0
    end = len(data)
    DOB = data.columns.get_loc("DOB")
    while i < end:
        year = col[i][:4]
        data.iloc[i, DOB] = year
        i += 1
    return data

def gender_update(data):
    col = data["Gender"]
    i = 0
    end = len(data)
    gender_col = data.columns.get_loc("Gender")
    while i < end:
        gender = 0
        if col[i] == 'f':
            gender = 1
        data.iloc[i, gender_col] = gender
        i += 1
    return data

def degree_update(data):
    col = data["Degree"]
    i = 0
    end = len(data)
    degree_col = data.columns.get_loc("Degree")
    while i < end:
        degree = 0
        cell_info = col[i]
        if cell_info == 'B.Tech/B.E.':
            degree = 0
        elif cell_info == 'M.Tech./M.E.':
            degree = 1
        elif cell_info == 'MCA':
            degree = 2
        elif cell_info == 'M.Sc. (Tech.)':
            degree = 3
        data.iloc[i, degree_col] = degree
        i += 1
    return data

def spec_update(data):
    col = data["Specialization"]
    spec_col = data.columns.get_loc("Specialization")
    spec_names = []
    for i in range(len(col)):
        el = data.iloc[i, spec_col]
        if el in spec_names:
            idx = spec_names.index(el)
            data.iloc[i, spec_col] = idx
        else:
            idx = len(spec_names)
            spec_names.append(el)
            data.iloc[i, spec_col] = idx
    return data

def collegeState_update(data):
    col = data["CollegeState"]
    collegeState_col = data.columns.get_loc("CollegeState")
    collegeState_names = []
    for i in range(len(col)):
        el = data.iloc[i, collegeState_col]
        if el in collegeState_names:
            idx = collegeState_names.index(el)
            data.iloc[i, collegeState_col] = idx
        else:
            idx = len(collegeState_names)
            collegeState_names.append(el)
            data.iloc[i, collegeState_col] = idx
    return data

def correct_values(data):
    column_names = []
    for col in data.columns:
        column_names.append(col)

    #print(data.dtypes)
    data['Gender'] = data['Gender'].astype('int64')
    data['DOB'] = data['DOB'].astype('int64')
    data['Degree'] = data['Degree'].astype('int64')
    data['Specialization'] = data['Specialization'].astype('int64')
    data['CollegeState'] = data['CollegeState'].astype('int64')

    for i in range(len(data)):
        cols = ['GraduationYear', 'Domain', 'ComputerProgramming', 'ElectronicsAndSemicon', 'ComputerScience']
        for j in range(len(cols)):
            col = data.columns.get_loc(cols[j])
            content = data.iloc[i, col]
            if j == 0:
                if content <= 0:
                    data.iloc[i, col] = None  
            else:
                if content < 0:
                    data.iloc[i, col] = None

    print(data.isnull().sum()/data.shape[0])
    data = data.fillna(-9999)
    return data



