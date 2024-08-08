import pkg_resources

# 获取所有已安装的包
installed_packages = pkg_resources.working_set

# 过滤掉本地路径引用
requirements = []
for package in installed_packages:
    if 'file://' in package.location or '://' in package.location:
        # 忽略本地路径引用或特定URL源
        continue
    requirements.append(f"{package.key}=={package.version}")

# 将结果写入 requirements.txt
with open('requirements.txt', 'w') as f:
    for requirement in requirements:
        f.write(requirement + '\n')

print("Requirements exported successfully to requirements.txt")
