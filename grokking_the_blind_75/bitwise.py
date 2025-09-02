def encode(strings):
    """
    Edcative.io code
    """
    encoded_string = ""

    for x in strings:
        encoded_string += length_to_bytes(x) + x

    return encoded_string


def decode(string):
    """
    Edcative.io code
    """
    i = 0
    decoded_string = []

    while i < len(string):
        length = bytes_to_length(string[i : i + 4])
        i += 4
        decoded_string.append(string[i : i + length])
        i += length

    return decoded_string


def length_to_bytes(x):
    """
    Edcative.io code
    """
    length = len(x)
    bytes = []

    for i in range(4):
        bytes.append(chr(length >> (i * 8)))

    bytes.reverse()
    bytes_string = "".join(bytes)

    return bytes_string


def bytes_to_length(bytes_string):
    """
    Edcative.io code
    """
    result = 0

    for c in bytes_string:
        result = result * 256 + ord(c)

    return result


def count_bits(n):
    count = 0
    while n != 0:
        if n & 1 == 1:
            count += 1
        n = n >> 1
    return count


def reverse_bits(n):
    res = 0
    while n > 0:
        res = res << 1
        next_value = n & 1
        res = res | next_value
        n = n >> 1
    return res


if __name__ == '__main__':
    encoded_string = encode(["I", "love", "educative"])
    print(encoded_string)
    decoded_string = decode(encoded_string)
    print(decoded_string)
