#include <iostream>
#include <set>
#include <algorithm>
#include <cstring>

using namespace std;

/*
int sudoku[9][9] = {

        {5, 3, 0,  0, 7, 0,  0, 0, 0},
        {6, 0, 0,  1, 9, 5,  0, 0, 0},
        {0, 9, 8,  0, 0, 0,  0, 6, 0},

        {8, 0, 0,  0, 6, 0,  0, 0, 3},
        {4, 0, 0,  8, 0, 3,  0, 0, 1},
        {7, 0, 0,  0, 2, 0,  0, 0, 6},

        {0, 6, 0,  0, 0, 0,  2, 8, 0},
        {0, 0, 0,  4, 1, 9,  0, 0, 5},
        {0, 0, 0,  0, 8, 0,  0, 7, 9}

};
*/

void print_sudoku(int sudoku[9][9]){
    int col_space = 3;
    int row_space = 3;
    int r = 1;
    int c = 1;
    for(int y = 0; y < 9; y++){

        for(int x = 0; x < 9; x++){
            cout << sudoku[y][x] << " ";
            if(c >= col_space) {
                c = 1;
                cout << "\t";
            }else{
                c++;
            }
        }

        if(r >= row_space){
            r = 1;
            cout << endl;
        }else{
            r++;
        }
        cout << endl;
    }
}

// Check if it's safe to place a number in the given cell
bool is_safe(int grid[9][9], int row, int col, int num) {
    for (int x = 0; x < 9; x++) {
        if (grid[row][x] == num || grid[x][col] == num ||
            grid[3 * (row / 3) + x / 3][3 * (col / 3) + x % 3] == num) {
            return false;
        }
    }
    return true;
}

// Solve the Sudoku (used for generating a complete grid)
bool solveSudoku(int grid[9][9], int row, int col) {
    if (row == 9 - 1 && col == 9) {
        return true;
    }
    if (col == 9) {
        row++;
        col = 0;
    }
    if (grid[row][col] > 0) {
        return solveSudoku(grid, row, col + 1);
    }
    for (int num = 1; num <= 9; num++) {
        if (is_safe(grid, row, col, num)) {
            grid[row][col] = num;
            if (solveSudoku(grid, row, col + 1)) {
                return true;
            }
        }
        grid[row][col] = 0;
    }
    return false;
}

// Generate a random Sudoku puzzle
void generate_sudoku(int grid[9][9], int numClues) {
    memset(grid, 0, sizeof(grid[0][0]) * 9 * 9);

    // Fill the diagonal 3x3 matrices
    for (int i = 0; i < 9; i += 3) {
        for (int j = 0; j < 9; j++) {
            int num = (rand() % 9) + 1;
            while (!is_safe(grid, i + (j / 3), i + (j % 3), num)) {
                num = (rand() % 9) + 1;
            }
            grid[i + (j / 3)][i + (j % 3)] = num;
        }
    }

    // Solve the partially filled grid to get a complete solution
    solveSudoku(grid, 0, 0);

    // Remove numbers to create the puzzle
    int count = 9 * 9 - numClues;
    while (count > 0) {
        int i = rand() % 9;
        int j = rand() % 9;
        if (grid[i][j] != 0) {
            grid[i][j] = 0;
            count--;
        }
    }
}

void print_set(set<int>* s, bool linebreak=true){
	set<int>::iterator it;
	cout << "set[ ";
	for(it = s->begin(); it != s->end(); ++it){
		cout << *it;
		if(it != s->end()){
			 cout << " ";
		}
	}
	cout << "]";
	if(linebreak) cout << endl;
}

int possibile[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
set<int>** make_sets(){
	set<int>** sets = new set<int>*[9];
	for(int i = 0; i < 9; i++){
		sets[i] = new set<int>(possibile,possibile+9);

	}

	return sets;
}

void delete_sets(set<int>** sets){
	for(int i = 0; i < 9; i++){
		delete sets[i];
	}
	delete sets;
}

set<int>** fetch_rows(int sudoku[9][9]){
	set<int>** rows = make_sets();
	for(int i = 0; i < 9; i++){
		for(int j = 0; j < 9; j++){
			rows[i]->erase(sudoku[i][j]);
		}
	}
	return rows;
}

set<int>** fetch_cols(int sudoku[9][9]){
	set<int>** cols = make_sets();
	for(int i = 0; i < 9; i++){
		for(int j = 0; j < 9; j++){
			cols[i]->erase(sudoku[j][i]);
		}
	}
	return cols;
}

set<int>** fetch_grids(int sudoku[9][9]){
	set<int>** grids = make_sets();

	for(int i = 0; i < 9; i++){
		for(int j = 0; j < 9; j++){
			int x = j % 3 + ((i % 3) * 3);
			int y = j / 3 + (((i / 3) % 3) * 3);
			grids[i]->erase(sudoku[y][x]);
		}
	}
	return grids;
}

int grid(int x, int y){
	return (x / 3) + (y / 3) * 3;
}

set<int> multi_intersection(const set<int>* a, const set<int>* b, const set<int>* c){
	const set<int>* sets[3] = {a, b, c};
	set<int> last_interesection = *a;
	set<int> current_interesection;
	for(int i = 1; i < 3; i++){
		set_intersection(last_interesection.begin(), last_interesection.end(),
						 sets[i]->begin(), sets[i]->end(),
						 inserter(current_interesection, current_interesection.begin()));
		swap(last_interesection, current_interesection);
		current_interesection.clear();
	}
	return last_interesection;
}

bool validate_sudoku(int sudoku[9][9]){
	set<int> check_backup(possibile, possibile+9);
	for(int y = 0; y < 9; y++){
		set<int> check = check_backup;
		for(int x = 0; x < 9; x++){
			check.erase(sudoku[y][x]);
		}
		if(check.size() != 0) return false;
	}

	for(int x = 0; x < 9; x++){
		set<int> check = check_backup;
		for(int y = 0; y < 9; y++){
			check.erase(sudoku[y][x]);
		}
		if(check.size() != 0) return false;
	}

	for(int i = 0; i < 9; i++){
		set<int> check = check_backup;
		for(int j = 0; j < 9; j++){
			int x = j % 3 + ((i % 3) * 3);
			int y = j / 3 + (((i / 3) % 3) * 3);
			check.erase(sudoku[y][x]);
		}
		if(check.size() != 0) return false;
	}
	return true;
}

bool solve_sudoku(int sudoku[9][9]){
	set<int>** rows = fetch_rows(sudoku);
	set<int>** cols = fetch_cols(sudoku);
	set<int>** grids = fetch_grids(sudoku);

    bool solvable = true;
    while(solvable) {
        solvable = false;

        vector<tuple<int, int, int>> options;

		for(int y = 0; y < 9; y++){
			for(int x = 0; x < 9; x++){
				if(sudoku[y][x] == 0){
                    int g = grid(x, y);
					set<int> could_be = multi_intersection(cols[x], rows[y], grids[g]);
                    options.emplace_back(could_be.size(), x, y);
					if(could_be.size() == 1) {
						int value = *could_be.begin();
						sudoku[y][x] = value;
						rows[y]->erase(value);
						cols[x]->erase(value);
						grids[g]->erase(value);
                        solvable = true;
                        continue;
                    }
				}
			}
		}

        if (solvable) continue;

        sort(options.begin(), options.end());

        for (const auto& option : options) {
            int size, x, y;
            tie(size, x, y) = option;

            int g = grid(x, y);
            set<int> could_be = multi_intersection(cols[x], rows[y], grids[g]);

            int duplicate_sudoku[9][9];
            std::copy(&sudoku[0][0], &sudoku[0][0] + 81, &duplicate_sudoku[0][0]);

            for (int value : could_be) {
                duplicate_sudoku[y][x] = value;
                if (solve_sudoku(duplicate_sudoku)) {
                    std::copy(&duplicate_sudoku[0][0], &duplicate_sudoku[0][0] + 81, &sudoku[0][0]);
                    solvable = true;
                }
            }
        }

        bool solved = true;
        for(int y = 0; y < 9; y++) {
            for (int x = 0; x < 9; x++) {
                if (sudoku[y][x] == 0) {
                    solved = false;
                }
            }
        }

        if (solved) {
            solvable = true;
            break;
        }
	}
	delete_sets(rows);
	delete_sets(cols);
	delete_sets(grids);

    return solvable;
}


int main(int argc, char** argv){

    for (int i = 0; i < 100; i++) {
        std::cout << i << std::endl;
        int sudoku[9][9];
        int num_clues = 32;

        generate_sudoku(sudoku, num_clues);
        solve_sudoku(sudoku);
        print_sudoku(sudoku);
        if (!validate_sudoku(sudoku)) {
            return 1;
        }
    }
}