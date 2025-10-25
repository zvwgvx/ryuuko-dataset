import pandas as pd

# Đọc file với skiprows=1 để bỏ qua dòng đầu tiên (header cũ)
df = pd.read_csv('teenslangs.csv', skiprows=1, header=None)

# Đặt tên cột theo header thực
df.columns = [
    'prompt', 'response', 'topic', 'primary_emotion',
    'secondary_emotions', 'emotional_intensity', 'tsun_level',
    'dere_level', 'context', 'memory_tags'
]

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())