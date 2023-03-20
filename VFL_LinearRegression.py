import numpy as np
from phe import paillier

class Client(object):
    def __init__(self,config):
        # 模型训练过程中产生的所有数据
        self.data = {}
        self.config = config
        self.other_clinet = {}
    def send_data(self,data,target_client):
        target_client.data.update(data)

class ClientA(Client):
    def __init__(self,X,config):
        super().__init__(config)
        self.X = X
        # 初始化参数
        self.weights = np.zeros(self.X.shape[1])
    # 计算u_a
    def compute_u_a(self):
        u_a = self.X.dot(self.weights)
        return u_a
    # 计算加密梯度
    def compute_encrypted_dL_a(self,encrypted_d):
        encrypted_dL_a = self.X.T.dot(encrypted_d) + self.config['lambda'] * self.weights
        return encrypted_dL_a
    # 做predict
    def predict(self,X_test):
        u_a = X_test.dot(self.weights)
        return u_a
    # 计算[[u_a]],[[L_a]]发送给B方
    def task_1(self,client_B_name):
        dt = self.data
        # 获取公钥
        assert 'public_key' in dt.keys(),"Error: 'public_key' from C in step 1 not receive successfully"
        public_key = dt['public_key']
        u_a = self.compute_u_a()
        encrypted_u_a = np.array([public_key.encrypt(x) for x in u_a])
        u_a_square = u_a ** 2
        L_a = 0.5*np.sum(u_a_square) + 0.5 * self.config['lambda'] * np.sum(self.weights**2)
        encrypted_L_a = public_key.encrypt(L_a)
        data_to_B = {'encrypted_u_a':encrypted_u_a,'encrypted_L_a':encrypted_L_a}
        self.send_data(data_to_B,self.other_clinet[client_B_name])
    # 计算加密梯度[[dL_a]]，加上随机数之后，发送给C
    def task_2(self,client_C_name):
        dt = self.data
        assert 'encrypted_d' in dt.keys(),"Error: 'encrypted_d' from B in step 1 not receive successfully"
        encrypted_d = dt['encrypted_d']
        encrypted_dL_a = self.compute_encrypted_dL_a(encrypted_d)
        mask = np.random.rand(len(encrypted_dL_a))
        encrypted_masked_dL_a = encrypted_dL_a + mask
        self.data.update({'mask':mask})
        data_to_C = {'encrypted_masked_dL_a':encrypted_masked_dL_a}
        self.send_data(data_to_C,self.other_clinet[client_C_name])
    # 获取解密后的masked梯度，减去mask，梯度下降更新
    def task_3(self):
        dt = self.data
        assert 'mask' in dt.keys(),"Error: 'mask' form A in step 2 not receive successfully"
        assert 'masked_dL_a' in dt.keys(), "Error: 'masked_dL_a' from C in step 1 not receive successfully"
        mask = dt['mask']
        masked_dL_a = dt['masked_dL_a']
        dL_a = masked_dL_a - mask
        # 注意这里的1/n
        self.weights = self.weights - self.config['lr'] * dL_a / len(self.X)
        print("A weights : {}".format(self.weights))

class ClientB(Client):
    def __init__(self,X,y,config):
        super().__init__(config)
        self.X = X
        self.y = y
        self.weights = np.zeros(self.X.shape[1])
    # 计算u_b
    def compute_u_b(self):
        u_b = self.X.dot(self.weights)
        return u_b
    # 计算加密梯度
    def compute_encrypted_dL_b(self,encrypted_d):
        encrypted_dL_b = self.X.T.dot(encrypted_d) + self.config['lambda'] * self.weights
        return encrypted_dL_b
    # 做predict
    def predict(self,X_test):
        u_b = X_test.dot(self.weights)
        return u_b
    # 计算[[d]] 发送给A方；计算[[L]]，发送给C方
    def task_1(self,client_A_name,client_C_name):
        dt = self.data
        assert 'encrypted_u_a' in dt.keys(),"Error: 'encrypted_u_a' from A in step 1 not receive successfully"
        encrypted_u_a = dt['encrypted_u_a']
        u_b = self.compute_u_b()
        z_b = u_b - self.y
        z_b_square = z_b**2
        encrypted_d = encrypted_u_a + z_b
        data_to_A = {'encrypted_d':encrypted_d}
        self.data.update({'encrypted_d':encrypted_d})
        assert 'encrypted_L_a' in dt.keys(),"Error,'encrypted_L_a' from A in step 1 not receive successfully"
        encrypted_L_a = dt['encrypted_L_a']
        L_b = 0.5 * np.sum(z_b_square) + 0.5 * self.config['lambda'] * np.sum(self.weights**2)
        L_ab = np.sum(encrypted_u_a * z_b)
        encrypted_L = encrypted_L_a + L_b + L_ab
        data_to_C = {'encrypted_L':encrypted_L}
        self.send_data(data_to_A,self.other_clinet[client_A_name])
        self.send_data(data_to_C, self.other_clinet[client_C_name])
    # 计算加密梯度[[dL_b]],mask之后发给C方
    def task_2(self,client_C_name):
        dt = self.data
        assert 'encrypted_d' in dt.keys(),"Error: 'encrypted_d' from B in step 1 not receive successfully"
        encrypted_d = dt['encrypted_d']
        encrypted_dL_b = self.compute_encrypted_dL_b(encrypted_d)
        mask = np.random.rand(len(encrypted_dL_b))
        encrypted_masked_dL_b = encrypted_dL_b + mask
        self.data.update({'mask':mask})
        data_to_C = {'encrypted_masked_dL_b':encrypted_masked_dL_b}
        self.send_data(data_to_C,self.other_clinet[client_C_name])
    # 获取解密后的梯度，解mask，模型更新
    def task_3(self):
        dt = self.data
        assert 'mask' in dt.keys(), "Error: 'mask' form B in step 2 not receive successfully"
        assert 'masked_dL_b' in dt.keys(), "Error: 'masked_dL_b' from C in step 1 not receive successfully"
        mask = dt['mask']
        masked_dL_b = dt['masked_dL_b']
        dL_b = masked_dL_b - mask
        self.weights = self.weights - self.config['lr'] * dL_b / len(self.X)
        print("B weights : {}".format(self.weights))

class ClientC(Client):
    def __init__(self,config):
        super().__init__(config)
        self.loss_history = []
        self.public_key = None
        self.private_key = None
    # 产生钥匙对，将公钥发送给A,B方
    def task_1(self,client_A_name,client_B_name):
        self.public_key,self.private_key = paillier.generate_paillier_keypair()
        data_to_AB = {'public_key':self.public_key}
        self.send_data(data_to_AB,self.other_clinet[client_A_name])
        self.send_data(data_to_AB, self.other_clinet[client_B_name])
    # 解密[[L]]、[[masked_dL_a]],[[masked_dL_b]]，分别发送给A、B
    def task_2(self,client_A_name,client_B_name):
        dt = self.data
        assert 'encrypted_L' in dt.keys(),"Error: 'encrypted_L' from B in step 2 not receive successfully"
        assert 'encrypted_masked_dL_b' in dt.keys(), "Error: 'encrypted_masked_dL_b' from B in step 2 not receive successfully"
        assert 'encrypted_masked_dL_a' in dt.keys(), "Error: 'encrypted_masked_dL_a' from A in step 2 not receive successfully"
        encrypted_L = dt['encrypted_L']
        encrypted_masked_dL_b = dt['encrypted_masked_dL_b']
        encrypted_masked_dL_a = dt['encrypted_masked_dL_a']
        L = self.private_key.decrypt(encrypted_L)
        print('*'*8,L,'*'*8)
        self.loss_history.append(L)
        masked_dL_b = np.array([self.private_key.decrypt(x) for x in encrypted_masked_dL_b])
        masked_dL_a = np.array([self.private_key.decrypt(x) for x in encrypted_masked_dL_a])
        data_to_A = {'masked_dL_a':masked_dL_a}
        data_to_B = {'masked_dL_b':masked_dL_b}
        self.send_data(data_to_A, self.other_clinet[client_A_name])
        self.send_data(data_to_B, self.other_clinet[client_B_name])
