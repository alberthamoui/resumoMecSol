import numpy as np

class Estrutura:
    def __init__(self, nos, incidencia, propriedades):
        self.nos = nos
        self.incidencia = incidencia
        self.propriedades = propriedades
        self.gdl = self._calcular_gdl()

    def _calcular_gdl(self):
        num_nos = len(self.nos)
        gdl = np.zeros((num_nos, 2), dtype=int)
        for i in range(num_nos):
            gdl[i] = [2 * i, 2 * i + 1]
        return gdl

    def matriz_rigidez_elemento(self, x1, y1, x2, y2):
        A = self.propriedades['A']
        E = self.propriedades['E']
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        k = (A * E / L) * np.array([
            [c*c, c*s, -c*c, -c*s],
            [c*s, s*s, -c*s, -s*s],
            [-c*c, -c*s, c*c, c*s],
            [-c*s, -s*s, c*s, s*s]
        ])
        return k
    
    def matriz_local(self, n_elemento):
        no1, no2 = self.incidencia[n_elemento]
        x1, y1 = self.nos[no1]
        x2, y2 = self.nos[no2]
        return self.matriz_rigidez_elemento(x1, y1, x2, y2)

class AnaliseEstrutural:
    def __init__(self, estrutura):
        self.estrutura = estrutura
        self.K_global = self._montar_matriz_rigidez_global()

    def _montar_matriz_rigidez_global(self):
        num_gdl = 2 * len(self.estrutura.nos)
        K_global = np.zeros((num_gdl, num_gdl))
        for elem in self.estrutura.incidencia:
            i, j = elem
            x1, y1 = self.estrutura.nos[i]
            x2, y2 = self.estrutura.nos[j]
            k = self.estrutura.matriz_rigidez_elemento(x1, y1, x2, y2)
            gdl_elem = np.hstack((self.estrutura.gdl[i], self.estrutura.gdl[j]))
            for a in range(4):
                for b in range(4):
                    K_global[gdl_elem[a], gdl_elem[b]] += k[a, b]
        return K_global

    def aplicar_condicoes_contorno(self, restricoes, F):
        for i, g in enumerate(restricoes['gdl']):
            self.K_global[g, :] = 0
            self.K_global[g, g] = 1
            F[g] = restricoes['valores'][i]
        return F

    def gauss_seidel(self, A, b, x0, tol=1e-10, max_iter=1000):
        n = len(b)
        x = x0.copy()
        for k in range(max_iter):
            x_new = np.zeros_like(x)
            for i in range(n):
                s1 = np.dot(A[i, :i], x_new[:i])
                s2 = np.dot(A[i, i + 1:], x[i + 1:])
                x_new[i] = (b[i] - s1 - s2) / A[i, i]
            if np.allclose(x, x_new, atol=tol):
                break
            x = x_new
        return x

    def resolver_sistema(self, F, restricoes):
        F = self.aplicar_condicoes_contorno(restricoes, F)
        x0 = np.zeros(len(F))
        deslocamentos = self.gauss_seidel(self.K_global, F, x0)
        return deslocamentos

    def calcular_tensoes(self, deslocamentos):
        tensoes = []
        for elem in self.estrutura.incidencia:
            i, j = elem
            x1, y1 = self.estrutura.nos[i]
            x2, y2 = self.estrutura.nos[j]
            L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            c = (x2 - x1) / L
            s = (y2 - y1) / L
            gdl_elem = np.hstack((self.estrutura.gdl[i], self.estrutura.gdl[j]))
            u_elem = deslocamentos[gdl_elem]
            deformacao = (1 / L) * np.dot([-c, -s, c, s], u_elem)
            tensao = self.estrutura.propriedades['E'] * deformacao
            tensoes.append(tensao)
        return tensoes

    def calcular_reacoes(self, deslocamentos, F):
        return np.dot(self.K_global, deslocamentos) - F

    def verificar_falha_tensao(self, tensoes):
        falha_tensao = []
        for tensao in tensoes:
            if tensao > self.estrutura.propriedades['T_rup_tensao']:
                falha_tensao.append("Falha por tração")
            elif tensao < -self.estrutura.propriedades['T_rup_compressao']:
                falha_tensao.append("Falha por compressão")
            else:
                falha_tensao.append("Sem falha")
        return falha_tensao

    def verificar_falha_flambagem(self, tensoes):
        falha_flambagem = []
        I = (self.estrutura.propriedades['A']**2) / 12  # Momento de inércia para seção retangular
        for tensao, elem in zip(tensoes, self.estrutura.incidencia):
            if tensao < 0:  # Só considerar elementos em compressão
                i, j = elem
                L = np.sqrt((self.estrutura.nos[j, 0] - self.estrutura.nos[i, 0])**2 + (self.estrutura.nos[j, 1] - self.estrutura.nos[i, 1])**2)
                P_cr = (np.pi**2 * self.estrutura.propriedades['E'] * I) / (L**2)  # Carga crítica de flambagem
                if abs(tensao * self.estrutura.propriedades['A']) > P_cr:
                    falha_flambagem.append("Falha por flambagem")
                else:
                    falha_flambagem.append("Sem falha por flambagem")
            else:
                falha_flambagem.append("Sem falha por flambagem")
        return falha_flambagem

# Dados de entrada
L = 1.0  # Comprimento de cada segmento

#coordenadas de todos os nós, uma vez cada
nos = np.array([
    [0 , 0], [L, 0], [2*L, 0], [3*L, 0], [4*L, 0], [5*L, 0], [6*L, 0], [7*L, 0], [8*L, 0],  # Nós inferiores
    [L, L], [2*L, L], [3*L, L], [4*L, L], [5*L, L], [6*L, L], [7*L, L]  # Nós superiores
])

#
incidencia = np.array([
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],  # Elementos horizontais inferiores
    [0, 9], [1, 9], [1, 10], [2, 10], [2, 11], [3, 11], [3, 12], [4, 12], [4, 13], [5, 13], [5, 14], [6, 14], [6, 15], [7, 15],  # Elementos diagonais
    [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15]  # Elementos horizontais superiores
])

propriedades = {
    'A': 6e-4,  # Área da seção transversal em m²
    'E': 200e9,  # Módulo de elasticidade em Pa (200 GPa)
    'T_rup_tensao': 400e6,  # Tensão de ruptura a tração em Pa
    'T_rup_compressao': 250e6,  # Tensão de ruptura a compressão em Pa
}


#sempre comecar os gdls do 0
#onde tem as restricoes (pinos e roletes)
restricoes = {
    'gdl': [0, 1, 17],  # Graus de liberdade restritos (nós 0 e 8)
    'valores': [0, 0, 0]  # Valores de deslocamento para os graus de liberdade restritos
}

#onde tem força externa no gdl
cargas = {
    'gdl': [21, 23, 25],  # Graus de liberdade verticais dos nós 11, 12 e 13
    'valores': [-100, -100, -100]  # Forças aplicadas de 1000 N cada, para baixo (negativo)
}

# Inicialização da estrutura e análise
estrutura = Estrutura(nos, incidencia, propriedades)
analise = AnaliseEstrutural(estrutura)

# Montagem do vetor de carga global
#dentro do np.zeros = quantidade de gdl
F = np.zeros(32)
for i, g in enumerate(cargas['gdl']):
    F[g] = cargas['valores'][i]

# Solução do sistema de equações
deslocamentos = analise.resolver_sistema(F, restricoes)

# Pós-processamento: cálculo das tensões e reações de apoio
tensoes = analise.calcular_tensoes(deslocamentos)
reacoes = analise.calcular_reacoes(deslocamentos, F)
falha_tensao = analise.verificar_falha_tensao(tensoes)
falha_flambagem = analise.verificar_falha_flambagem(tensoes)

# Resultados
print("Deslocamentos nos nós (em m):")
print(deslocamentos)
print("\nTensões em cada elemento (em Pa):")
print(tensoes)
print("\nReações de apoio nos nós com restrição (em N):")
print(reacoes[restricoes['gdl']])
print("\nmatriz global")
print(analise.K_global)
# print("\nAnálise de falha por tensão:")
# print(falha_tensao)
# print("\nAnálise de falha por flambagem:")
# print(falha_flambagem)
