#include <iostream>
#include <set>
#include <algorithm>

using namespace std;

int suduko[9][9] = {

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
	if(linebreak)
		 cout << endl;
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

set<int>** fetch_rows(int suduko[9][9]){
	set<int>** rows = make_sets();
	for(int i = 0; i < 9; i++){
		for(int j = 0; j < 9; j++){
			rows[i]->erase(suduko[i][j]);
		}
	}


	return rows;

}

set<int>** fetch_cols(int suduko[9][9]){
	set<int>** cols = make_sets();
	for(int i = 0; i < 9; i++){
		for(int j = 0; j < 9; j++){
			cols[i]->erase(suduko[j][i]);
		}
	}
	return cols;
}

set<int>** fetch_grids(int suduko[9][9]){
	set<int>** grids = make_sets();

	for(int i = 0; i < 9; i++){
		for(int j = 0; j < 9; j++){
			int x = j % 3 + ((i % 3) * 3);
			int y = j / 3 + (((i / 3) % 3) * 3);
			grids[i]->erase(suduko[y][x]);
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



bool validate_suduko(int suduko[9][9]){


	set<int> check_backup(possibile, possibile+9);
	for(int y = 0; y < 9; y++){
		set<int> check = check_backup;
		for(int x = 0; x < 9; x++){
			check.erase(suduko[y][x]);
		}
		if(check.size() != 0) return false;
	}

	for(int x = 0; x < 9; x++){
		set<int> check = check_backup;
		for(int y = 0; y < 9; y++){
			check.erase(suduko[y][x]);
		}
		if(check.size() != 0) return false;
	}

	for(int i = 0; i < 9; i++){
		set<int> check = check_backup;
		for(int j = 0; j < 9; j++){
			int x = j % 3 + ((i % 3) * 3);
			int y = j / 3 + (((i / 3) % 3) * 3);
			check.erase(suduko[y][x]);
		}
		if(check.size() != 0) return false;
	}

	return true;
}
void solve_suduko(int suduko[9][9]){
	set<int>** rows = fetch_rows(suduko);
	set<int>** cols = fetch_cols(suduko);
	set<int>** grids = fetch_grids(suduko);

	bool solved = false;

	while(!solved){
		solved = true;
		for(int y = 0; y < 9; y++){
			for(int x = 0; x < 9; x++){
				int g = grid(x, y);

				if(suduko[y][x] == 0){
					set<int> could_be = multi_intersection(cols[x], rows[y], grids[g]);
					if(could_be.size() == 1){
						int value = *could_be.begin();
						suduko[y][x] = value;
						rows[y]->erase(value);
						cols[x]->erase(value);
						grids[g]->erase(value);
					} else if(could_be.size() > 1){
						solved = false;
					}
				}
			}
		}
	}



	delete_sets(rows);
	delete_sets(cols);
	delete_sets(grids);

}

void print_suduko(int suduko[9][9]){
	int col_space = 3;
	int row_space = 3;
	int r = 1;
	int c = 1;
	for(int y = 0; y < 9; y++){

		for(int x = 0; x < 9; x++){
			cout << suduko[y][x] << " ";
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

int main(int argc, char** argv){

	print_suduko(suduko);
	solve_suduko(suduko);

	if(validate_suduko(suduko)){
		cout << endl << "Success!" << endl;
		print_suduko(suduko);
	}
	return 0;
}