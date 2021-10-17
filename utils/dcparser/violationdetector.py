from string import Template
import pandas as pd
import pandasql as psql

unary_template = Template('SELECT * FROM $table as t1 WHERE $cond')
multi_template = Template('SELECT * FROM $table as t1 WHERE $cond1 $c EXISTS (SELECT * FROM $table as t2 WHERE $cond2)')


class ViolationDetector:
    def __init__(self, dataset, constraints):
        self.ds = dataset
        self.constraints = constraints

    def detect_noisy_cells(self):
        data = self.ds
        table = 'data'
        queries = []
        attrs = []
        predicates = []
        for c in self.constraints:
            q = self.to_sql(table, c)
            queries.append(q)
            attrs.append(c.components)
            predicates.append(c.predicates)

        results = []
        for query in queries:
            results.append(psql.sqldf(query, locals()))

        # Generate final output
        errors = []
        for i in range(len(attrs)):
            res = results[i]
            attr_list = attrs[i]
            predicates_list = predicates[i]
            tmp_df = self.get_output(res, attr_list)
            errors.append(tmp_df)
        errors_df = pd.concat(errors, ignore_index=True).drop_duplicates().reset_index(drop=True)
        return errors_df

    def to_sql(self, tbl, c):
        # Check tuples in constraint
        unary = len(c.tuple_names)==1
        if unary:
            query = self.gen_unary_query(tbl, c)
        else:
            query = self.gen_mult_query(tbl, c)
        return query

    def gen_unary_query(self, tbl, c):
        query = unary_template.substitute(table=tbl, cond=c.cnf_form)
        return query

    def gen_mult_query(self, tbl, c):
        # Iterate over constraint predicates to identify cond1 and cond2
        cond1_preds = []
        cond2_preds = []
        for pred in c.predicates:
            if 't1' in pred.cnf_form:
                if 't2' in pred.cnf_form:
                    cond2_preds.append(pred.cnf_form)
                else:
                    cond1_preds.append(pred.cnf_form)
            elif 't2' in pred.cnf_form:
                cond2_preds.append(pred.cnf_form)
            else:
                raise Exception("ERROR in violation detector. Cannot ground mult-tuple template.")
        cond1 = " AND ".join(cond1_preds)
        cond2 = " AND ".join(cond2_preds)
        a = ','.join(c.components)
        a = []
        for b in c.components:
            a.append("'"+b+"'")
        a = ','.join(a)
        if cond1 != '':
            query = multi_template.substitute(table=tbl, cond1=cond1, c='AND', cond2=cond2)
        else:
            query = multi_template.substitute(table=tbl, cond1=cond1, c='', cond2=cond2)
        return query

    def get_output(self, res, attr_list):
        errors = []
        for index, row in res.iterrows():
            errors.append({'index': int(row['index']), 'column': attr_list[-1], 'attribute': str(attr_list)})
        error_df = pd.DataFrame(data=errors)
        return error_df
