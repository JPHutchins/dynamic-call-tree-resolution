// Copyright (c) 2026 JP Hutchins
// SPDX-License-Identifier: MIT

#include "gcc-plugin.h"
#include "plugin-version.h"
#include "tree.h"
#include "tree-pass.h"
#include "context.h"
#include "function.h"
#include "basic-block.h"
#include "gimple.h"
#include "gimple-iterator.h"
#include "gimple-pretty-print.h"
#include "ssa.h"
#include "tree-pretty-print.h"

int plugin_is_GPL_compatible;

namespace {

pass_data const descriptors_data = {
	GIMPLE_PASS,
	"descriptors",
	OPTGROUP_NONE,
	TV_NONE,
	PROP_ssa,
	0,
	0,
	0,
	0,
};

char const * symbol(tree const decl) {
	return IDENTIFIER_POINTER(DECL_ASSEMBLER_NAME(decl));
}

char const * named(tree const name) {
	return (
		name == NULL_TREE ? "<anonymous>" :
		TREE_CODE(name) == TYPE_DECL ? IDENTIFIER_POINTER(DECL_NAME(name)) :
		IDENTIFIER_POINTER(name)
	);
}

char const * declared(tree const decl) {
	return DECL_P(decl) && DECL_NAME(decl) ? IDENTIFIER_POINTER(DECL_NAME(decl)) : "<expression>";
}

char const * record_kind(tree const record) {
	return TREE_CODE(record) == UNION_TYPE ? "union" : "struct";
}

void field(tree const reference) {
	tree const record = TYPE_MAIN_VARIANT(TREE_TYPE(TREE_OPERAND(reference, 0)));
	tree const member = TREE_OPERAND(reference, 1);
	printf(
		"%s %s\t%s",
		record_kind(record),
		named(TYPE_NAME(record)),
		DECL_NAME(member) ? IDENTIFIER_POINTER(DECL_NAME(member)) : "<anonymous>"
	);
}

void location(gimple const * const stmt) {
	expanded_location const place = expand_location(gimple_location(stmt));
	printf(
		"\t%s\t%s\t%d\t%d",
		symbol(current_function_decl),
		place.file ? place.file : "<unknown>",
		place.line,
		place.column
	);
}

tree loaded(tree const value) {
	gimple * const definition = TREE_CODE(value) == SSA_NAME ? SSA_NAME_DEF_STMT(value) : NULL;
	return (
		definition && gimple_assign_single_p(definition) ? gimple_assign_rhs1(definition) :
		NULL_TREE
	);
}

bool parameter(tree const value) {
	return (
		TREE_CODE(value) == SSA_NAME &&
		SSA_NAME_IS_DEFAULT_DEF(value) &&
		SSA_NAME_VAR(value) &&
		TREE_CODE(SSA_NAME_VAR(value)) == PARM_DECL
	);
}

void describe(tree const value) {
	tree const source = loaded(value);
	if (TREE_CODE(value) == ADDR_EXPR && TREE_CODE(TREE_OPERAND(value, 0)) == FUNCTION_DECL) {
		printf("\tfunction\t%s", symbol(TREE_OPERAND(value, 0)));
	} else if (integer_zerop(value)) {
		printf("\tnull\t0");
	} else if (parameter(value)) {
		printf("\tparameter\t%s", IDENTIFIER_POINTER(DECL_NAME(SSA_NAME_VAR(value))));
	} else if (source != NULL_TREE && TREE_CODE(source) == COMPONENT_REF) {
		printf("\tfield\t");
		field(source);
	} else if (source != NULL_TREE && TREE_CODE(source) == VAR_DECL) {
		printf("\tvariable\t%s", symbol(source));
	} else if (source != NULL_TREE && TREE_CODE(source) == ARRAY_REF) {
		printf("\tarray\t%s", declared(TREE_OPERAND(source, 0)));
	} else {
		printf("\tother\t");
		print_generic_expr(stdout, TREE_TYPE(value), TDF_NOUID);
	}
}

void site(gimple * const stmt) {
	printf("site");
	location(stmt);
	describe(gimple_call_fn(stmt));
	printf("\n");
}

bool function_pointer(tree const type) {
	return POINTER_TYPE_P(type) && TREE_CODE(TREE_TYPE(type)) == FUNCTION_TYPE;
}

void store(gimple * const stmt, tree const target) {
	printf("store");
	location(stmt);
	if (TREE_CODE(target) == COMPONENT_REF) {
		printf("\tfield\t");
		field(target);
	} else {
		printf("\tvariable\t%s", symbol(target));
	}
	if (gimple_assign_single_p(stmt)) {
		describe(gimple_assign_rhs1(stmt));
	} else {
		printf("\tother\t");
		print_generic_expr(stdout, TREE_TYPE(target), TDF_NOUID);
	}
	printf("\n");
}

struct descriptors : gimple_opt_pass {
	descriptors(gcc::context * const context) : gimple_opt_pass(descriptors_data, context) {}

	unsigned int execute(function * const fun) final override {
		basic_block block;
		FOR_EACH_BB_FN(block, fun) {
			for (
				gimple_stmt_iterator iterator = gsi_start_bb(block);
				!gsi_end_p(iterator);
				gsi_next(&iterator)
			) {
				gimple * const stmt = gsi_stmt(iterator);
				if (
					is_gimple_call(stmt) &&
					!gimple_call_internal_p(stmt) &&
					gimple_call_fndecl(stmt) == NULL_TREE
				) {
					site(stmt);
				}
				tree const target = gimple_get_lhs(stmt);
				if (
					target != NULL_TREE &&
					(TREE_CODE(target) == COMPONENT_REF || TREE_CODE(target) == VAR_DECL) &&
					function_pointer(TREE_TYPE(target))
				) {
					store(stmt, target);
				}
			}
		}
		return 0;
	}
};

}

int plugin_init(plugin_name_args * const info, plugin_gcc_version * const version) {
	if (!plugin_default_version_check(version, &gcc_version)) {
		return 1;
	}
	register_pass_info pass = {new descriptors(g), "alias", 1, PASS_POS_INSERT_AFTER};
	register_callback(info->base_name, PLUGIN_PASS_MANAGER_SETUP, NULL, &pass);
	return 0;
}
