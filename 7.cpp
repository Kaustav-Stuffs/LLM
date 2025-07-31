def isValid(s: str) -> bool:
    stack = []
    closing_to_opening = {')': '(', '}': '{', ']': '['}
   
    for char in s:
        if char in closing_to_opening.values():
            stack.append(char)
        elif char in closing_to_opening:
            if not stack:
                return False
            last_opening = stack.pop()
            if last_opening != closing_to_opening[char]:
                return False
        else:
            return False
   
    return not stack
print(isValid("()[]{}"))  
print(isValid("(]"))      
print(isValid("([{}])"))  
print(isValid("({[)]")) 
