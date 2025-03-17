from common.layout_display import LayoutDisplayMixin
import random
import numpy as np

class ParticleSwarm(LayoutDisplayMixin):
    def __init__(self, num_particles, num_iterations, dim, sheet_width, sheet_height, recortes_disponiveis):
        super().__init__()
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dim = dim
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.initial_layout = recortes_disponiveis
        self.optimized_layout = None
        self.particles = []
        self.velocities = []
        self.pbest = []
        self.gbest = None
        self.gbest_fitness = float('-inf')
        print("Particle Swarm Optimization Initialized.")

    def initialize_particles(self):
        """
        Inicializa as partículas com posições mais inteligentes.
        """
        for _ in range(self.num_particles):
            particle = []
            x_offset, y_offset = 0, 0  # Deslocamento para agrupar peças

            for recorte in self.initial_layout:
                if recorte['tipo'] == 'retangular' or recorte['tipo'] == 'diamante':
                    x = x_offset + random.uniform(0, 10)  # Pequena variação
                    y = y_offset + random.uniform(0, 10)
                    x_offset += recorte['largura'] + 5  # Espaçamento entre peças
                    if x_offset > self.sheet_width:
                        x_offset = 0
                        y_offset += recorte['altura'] + 5
                elif recorte['tipo'] == 'circular':
                    x = x_offset + random.uniform(0, 10)
                    y = y_offset + random.uniform(0, 10)
                    x_offset += 2 * recorte['r'] + 5
                    if x_offset > self.sheet_width:
                        x_offset = 0
                        y_offset += 2 * recorte['r'] + 5

                particle.append({
                    'x': x, 'y': y,
                    'rotacao': recorte.get('rotacao', 0),
                    'tipo': recorte['tipo'],
                    'largura': recorte.get('largura', 0),
                    'altura': recorte.get('altura', 0),
                    'r': recorte.get('r', 0)
                })

            self.particles.append(particle)
            self.velocities.append([{'x': random.uniform(-1, 1), 'y': random.uniform(-1, 1)} for _ in range(self.dim)])
            self.pbest.append(particle)
            
    def evaluate_particles(self):
        """
        Avalia cada partícula com base na função de fitness.
        """
        for i, particle in enumerate(self.particles):
            fitness = self.calculate_fitness(particle)
            if fitness > self.calculate_fitness(self.pbest[i]):
                self.pbest[i] = particle
            if fitness > self.gbest_fitness:
                self.gbest = particle
                self.gbest_fitness = fitness

    def update_velocity(self):
        """
        Atualiza a velocidade de cada partícula.
        """
        w = 0.9 - (0.5 * (self.iteration / self.num_iterations))  # Inércia decrescente
        c1 = 2.5 - (self.iteration / self.num_iterations)
        c2 = 1.5 + (self.iteration / self.num_iterations)

        for i in range(self.num_particles):
            for j in range(self.dim):
                r1 = random.random()
                r2 = random.random()
                self.velocities[i][j]['x'] = w * self.velocities[i][j]['x'] + \
                    c1 * r1 * (self.pbest[i][j]['x'] - self.particles[i][j]['x']) + \
                    c2 * r2 * (self.gbest[j]['x'] - self.particles[i][j]['x'])
                self.velocities[i][j]['y'] = w * self.velocities[i][j]['y'] + \
                    c1 * r1 * (self.pbest[i][j]['y'] - self.particles[i][j]['y']) + \
                    c2 * r2 * (self.gbest[j]['y'] - self.particles[i][j]['y'])
    def check_overlap(self, recorte1, recorte2):
        """
        Verifica se dois recortes se sobrepõem.
        """
        if recorte1['tipo'] in ['retangular', 'diamante'] and recorte2['tipo'] in ['retangular', 'diamante']:
            # Verifica sobreposição para retângulos e diamantes
            return not (recorte1['x'] + recorte1['largura'] < recorte2['x'] or
                        recorte1['x'] > recorte2['x'] + recorte2['largura'] or
                        recorte1['y'] + recorte1['altura'] < recorte2['y'] or
                        recorte1['y'] > recorte2['y'] + recorte2['altura'])
        elif recorte1['tipo'] == 'circular' and recorte2['tipo'] == 'circular':
            # Verifica sobreposição para círculos
            distance = np.sqrt((recorte1['x'] - recorte2['x'])**2 + (recorte1['y'] - recorte2['y'])**2)
            return distance < (recorte1['r'] + recorte2['r'])
        elif (recorte1['tipo'] in ['retangular', 'diamante'] and recorte2['tipo'] == 'circular') or \
             (recorte1['tipo'] == 'circular' and recorte2['tipo'] in ['retangular', 'diamante']):
            # Verifica sobreposição entre círculos e retângulos/diamantes
            if recorte1['tipo'] == 'circular':
                circle, rect = recorte1, recorte2
            else:
                circle, rect = recorte2, recorte1
            # Verifica se o círculo está dentro do retângulo
            closest_x = max(rect['x'], min(circle['x'], rect['x'] + rect['largura']))
            closest_y = max(rect['y'], min(circle['y'], rect['y'] + rect['altura']))
            distance = np.sqrt((circle['x'] - closest_x)**2 + (circle['y'] - closest_y)**2)
            return distance < circle['r']
        return False

    def calculate_proximity(self, recorte):
        """
        Calcula a proximidade de um recorte em relação ao canto superior esquerdo da chapa.
        Quanto mais próximo, maior o bônus.
        """
        return 1 / (1 + recorte['x'] + recorte['y'])

    def update_velocity(self, iteration):
        """
        Atualiza a velocidade de cada partícula, ajustando dinamicamente os pesos para evitar convergência prematura.
        """
        w_max, w_min = 0.9, 0.4  # Ajuste mais lento do peso de inércia
        w = w_max - (iteration / self.num_iterations) * (w_max - w_min)
        
        c1 = 2.5 - (iteration / self.num_iterations)  # Foco cognitivo
        c2 = 1.5 + (iteration / self.num_iterations)  # Foco social

        V_max, V_min = 5, -5  # Clamping da velocidade para evitar passos muito grandes

        for i in range(self.num_particles):
            for j in range(self.dim):
                r1, r2 = random.random(), random.random()

                # Atualiza velocidades com limites de mínimo e máximo
                self.velocities[i][j]['x'] = np.clip(
                    w * self.velocities[i][j]['x'] + 
                    c1 * r1 * (self.pbest[i][j]['x'] - self.particles[i][j]['x']) + 
                    c2 * r2 * (self.gbest[j]['x'] - self.particles[i][j]['x']),
                    V_min, V_max
                )
                self.velocities[i][j]['y'] = np.clip(
                    w * self.velocities[i][j]['y'] + 
                    c1 * r1 * (self.pbest[i][j]['y'] - self.particles[i][j]['y']) + 
                    c2 * r2 * (self.gbest[j]['y'] - self.particles[i][j]['y']),
                    V_min, V_max
                )

                # Adiciona mutação ocasional para evitar estagnação
                if random.random() < 0.1:  # 10% de chance de mutação
                    self.velocities[i][j]['x'] += np.random.normal(0, 1)  # Ruído Gaussiano
                    self.velocities[i][j]['y'] += np.random.normal(0, 1)

    
    def calculate_fitness(self, particle):
        """
        Calcula o fitness de uma partícula.
        Quanto menos sobreposições e mais compacto o layout, maior será o fitness.
        """
        overlap_penalty = 0
        proximity_bonus = 0
        total_area = 0  # Área total ocupada pelas peças

        for i in range(len(particle)):
            for j in range(i + 1, len(particle)):
                if self.check_overlap(particle[i], particle[j]):
                    overlap_penalty += 20  # Penalize mais agressivamente

            # Calcula a área ocupada pela peça
            if particle[i]['tipo'] in ['retangular', 'diamante']:
                total_area += particle[i]['largura'] * particle[i]['altura']
            elif particle[i]['tipo'] == 'circular':
                total_area += np.pi * (particle[i]['r'] ** 2)

            # Bônus por proximidade ao canto superior esquerdo
            proximity_bonus += 1 / (1 + particle[i]['x'] + particle[i]['y'])

        # Fitness é uma combinação de:
        # - Penalização por sobreposição
        # - Bônus por proximidade
        # - Maximização da área ocupada (para evitar espaços vazios)
        fitness = -overlap_penalty + proximity_bonus + (total_area / (self.sheet_width * self.sheet_height))
        return fitness
    
    def update_position(self):
        """
        Atualiza a posição de cada partícula com base na velocidade.
        Adiciona uma pequena mutação para evitar convergência prematura.
        """
        for i in range(self.num_particles):
            for j in range(self.dim):
                self.particles[i][j]['x'] += self.velocities[i][j]['x'] + random.uniform(-0.1, 0.1)
                self.particles[i][j]['y'] += self.velocities[i][j]['y'] + random.uniform(-0.1, 0.1)

                # Garante que as posições estejam dentro dos limites da chapa
                if self.particles[i][j]['tipo'] in ['retangular', 'diamante']:
                    self.particles[i][j]['x'] = max(0, min(self.sheet_width - self.particles[i][j]['largura'], self.particles[i][j]['x']))
                    self.particles[i][j]['y'] = max(0, min(self.sheet_height - self.particles[i][j]['altura'], self.particles[i][j]['y']))
                elif self.particles[i][j]['tipo'] == 'circular':
                    self.particles[i][j]['x'] = max(self.particles[i][j]['r'], min(self.sheet_width - self.particles[i][j]['r'], self.particles[i][j]['x']))
                    self.particles[i][j]['y'] = max(self.particles[i][j]['r'], min(self.sheet_height - self.particles[i][j]['r'], self.particles[i][j]['y']))
   
    def get_best_solution(self):
        """
        Retorna a melhor solução encontrada.
        """
        return self.gbest

    def run(self):
        """
        Executa o algoritmo PSO e salva os melhores fitness para análise.
        """
        self.initialize_particles()
        fitness_history = []  # Lista para armazenar a evolução do fitness

        for iteration in range(self.num_iterations):
            self.evaluate_particles()
            self.update_velocity(iteration)
            self.update_position()

            # Armazena o fitness global para análise
            fitness_history.append(self.gbest_fitness)

            if (iteration + 1) % 50 == 0:
                print(f"Iteração {iteration + 1}/{self.num_iterations} - Melhor Fitness Global: {self.gbest_fitness:.6f}")

        self.optimized_layout = self.get_best_solution()
        return self.optimized_layout  # Retorna apenas o layout otimizado


    def get_best_solution(self):
        """
        Retorna a melhor solução encontrada.
        """
        return self.gbest  # Retorna a melhor partícula (lista de recortes)

    def optimize_and_display(self):
        """
        Exibe o layout inicial, executa a otimização e exibe o layout otimizado.
        """
        # Exibe o layout inicial
        self.display_layout(self.initial_layout, title="Initial Layout - Particle Swarm")

        # Executa a otimização
        self.optimized_layout = self.run()

        # Exibe o layout otimizado
        if self.optimized_layout:
            self.display_layout(self.optimized_layout, title="Optimized Layout - Particle Swarm")
        else:
            print("Nenhum layout otimizado foi encontrado.")
        return self.optimized_layout