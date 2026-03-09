def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    new_values = values.copy()

    #Duyệt qua từng trạng thái s=i
    for i in range(len(values)):
        q_values = []

        #Duyệt qua k hành động a=k
        for k in range(len(transitions[i])):
            #Tính kì vọng của giá trị tương lai
            expected_values = sum(p * v for p, v in zip(transitions[i][k], values))

            #Tính q_value cho hành động k
            q_value = rewards[i][k] + gamma * expected_values
            q_values.append(q_value)

        #Lấy giá trị lớn nhất trong trạng thái đó
        new_values[i] = max(q_values)
    
    return new_values