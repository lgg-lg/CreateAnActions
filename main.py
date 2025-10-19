import pandas as pd
import numpy as np
import os

# 可选：如果你想要绘图
# import matplotlib.pyplot as plt
# import seaborn as sns

def main():
    # 1. 检查文件是否存在
    file_path = os.path.join("Data", "clean_titanic.csv")
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return

    # 2. 加载数据
    df = pd.read_csv(file_path)
    print("✅ 数据加载成功！")
    print(f"数据形状: {df.shape}")
    print("\n前5行数据:")
    print(df.head())

    # 3. 基本信息
    print("\n" + "="*50)
    print("📊 数据基本信息:")
    print(df.info())

    # 4. 检查缺失值（即使叫 clean，也建议检查）
    print("\n" + "="*50)
    print("🔍 缺失值统计:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "无缺失值")

    # 5. 描述性统计
    print("\n" + "="*50)
    print("📈 数值型变量描述性统计:")
    print(df.describe())

    # 6. 分类变量统计
    print("\n" + "="*50)
    print("🧮 分类变量统计 (如 Sex, Embarked, Pclass):")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in ['Sex', 'Embarked', 'Pclass']:
        if col in df.columns:
            print(f"\n{col} 的频数分布:")
            print(df[col].value_counts(dropna=False))

    # 7. 生存率分析
    print("\n" + "="*50)
    print("💡 生存率分析:")
    survival_rate = df['Survived'].mean()
    print(f"总体生存率: {survival_rate:.2%}")

    # 按性别分组的生存率
    if 'Sex' in df.columns:
        sex_survival = df.groupby('Sex')['Survived'].mean()
        print("\n按性别分组的生存率:")
        print(sex_survival)

    # 按舱位等级分组的生存率
    if 'Pclass' in df.columns:
        pclass_survival = df.groupby('Pclass')['Survived'].mean()
        print("\n按舱位等级 (Pclass) 分组的生存率:")
        print(pclass_survival)

    # 8. 年龄与生存关系（使用 NumPy 进行简单计算）
    if 'Age' in df.columns:
        print("\n" + "="*50)
        print("👴 年龄与生存关系:")
        survived_ages = df[df['Survived'] == 1]['Age'].dropna()
        not_survived_ages = df[df['Survived'] == 0]['Age'].dropna()

        print(f"幸存者平均年龄: {np.mean(survived_ages):.2f} 岁")
        print(f"未幸存者平均年龄: {np.mean(not_survived_ages):.2f} 岁")

    # 9. （可选）保存分析结果到 CSV
    # summary = df.groupby(['Pclass', 'Sex'])['Survived'].agg(['mean', 'count']).round(4)
    # summary.to_csv('./data/survival_summary.csv')
    # print("\n✅ 分组生存率已保存至 ./data/survival_summary.csv")

    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()
