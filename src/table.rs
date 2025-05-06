pub fn print_table(rows: Vec<Vec<String>>) {
    let mut widths = Vec::<usize>::new();

    for row in &rows {
        for (i, column) in row.iter().enumerate() {
            if widths.len() > i {
                widths[i] = std::cmp::max(widths[i], column.len());
            } else {
                widths.push(column.len());
            }
        }
    }

    let max_width = 1 + widths.iter().sum::<usize>() + 3 * widths.len();

    for (i, row) in rows.iter().enumerate() {
        if i == 0 {
            println!("{}", "-".repeat(max_width));
        }

        print!("| ");

        for (i, column) in row.iter().enumerate() {
            print!("{:1$} | ", column, widths[i]);
        }

        println!();

        if i == 0 || i == rows.len() - 1 {
            println!("{}", "-".repeat(max_width));
        }
    }
}
