import pandas as pd

file_path = 'D:/Desktop/code/waveform_design/python/anti_jam_waveform/MARL-Algorithms-master/common/Ber_data.csv'
df = pd.read_csv(file_path)

pattern = df['Bpsk-1/2']
bpsk_1_2_ber = pattern[0:400]
pattern = df['Qpsk-1/2']
qpsk_1_2_ber = pattern[0:400]
pattern = df['Qpsk-3/4']
qpsk_3_4_ber = pattern[0:400]
pattern = df['16Qam-1/2']
qam16_1_2_ber = pattern[0:400]
pattern = df['16Qam-3/4']
qam16_3_4_ber = pattern[0:400]
pattern = df['64Qam-2/3']
qam64_2_3_ber = pattern[0:400]
pattern = df['64Qam-3/4']
qam64_3_4_ber = pattern[0:400]
pattern = df['64Qam-5/6']
qam64_5_6_ber = pattern[0:400]
pattern = df['256Qam-3/4']
qam256_3_4_ber = pattern[0:400]
pattern = df['256Qam-3/4']
qam256_5_6_ber = pattern[0:400]

