class altitude_queue:

    def __init__(self):
        self.row = []
        self.length = 0
        self.added = 0
        self.avg = 0
        self.last_avg = 0

    def add(self, value):
        self.row.append(value)
        self.length += 1

        self.added += value
        if self.length > 10:
            self.added -= self.row[0]
            self.row = self.row[1:]
            self.length -= 1
        
        self.last_avg = self.avg
        self.avg = self.added / self.length
        return self.avg
        

    
    def get_avg(self):
        return self.avg
    

    def is_increasing(self):
        return self.last_avg < self.avg
    
    def is_decreasing(self):
        return self.last_avg > self.avg
    
    def change_of_height(self):
        return abs(self.last_avg - self.avg)

