	.text
	.file	"LLVMDialectModule"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "../mlir/test/Examples/Toy/Ch6" "dsp_op_delay_simple.py"
	.loc	1 1 0                           # dsp_op_delay_simple.py:1:0
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	.loc	1 31 11 prologue_end            # dsp_op_delay_simple.py:31:11
	movl	$80, %edi
	callq	malloc@PLT
	movq	%rax, %rbx
	xorl	%ebp, %ebp
	.loc	1 30 11                         # dsp_op_delay_simple.py:30:11
	movl	$80, %edi
	callq	malloc@PLT
	movq	%rax, %r14
	.loc	1 28 11                         # dsp_op_delay_simple.py:28:11
	movl	$8, %edi
	callq	malloc@PLT
	movq	%rax, %r15
	.loc	1 27 11                         # dsp_op_delay_simple.py:27:11
	movl	$8, %edi
	callq	malloc@PLT
	movq	%rax, %r12
	.loc	1 14 11                         # dsp_op_delay_simple.py:14:11
	movl	$80, %edi
	callq	malloc@PLT
	movq	%rax, %r13
	movabsq	$4621819117588971520, %rax      # imm = 0x4024000000000000
	movq	%rax, (%r13)
	movabsq	$4626322717216342016, %rax      # imm = 0x4034000000000000
	movq	%rax, 8(%r13)
	movabsq	$4629137466983448576, %rax      # imm = 0x403E000000000000
	movq	%rax, 16(%r13)
	movabsq	$4630826316843712512, %rax      # imm = 0x4044000000000000
	movq	%rax, 24(%r13)
	movabsq	$4632233691727265792, %rax      # imm = 0x4049000000000000
	movq	%rax, 32(%r13)
	movabsq	$4633641066610819072, %rax      # imm = 0x404E000000000000
	movq	%rax, 40(%r13)
	movabsq	$4634626229029306368, %rax      # imm = 0x4051800000000000
	movq	%rax, 48(%r13)
	movabsq	$4635329916471083008, %rax      # imm = 0x4054000000000000
	movq	%rax, 56(%r13)
	movabsq	$4636033603912859648, %rax      # imm = 0x4056800000000000
	movq	%rax, 64(%r13)
	movabsq	$4636737291354636288, %rax      # imm = 0x4059000000000000
	movq	%rax, 72(%r13)
	movabsq	$4611686018427387904, %rax      # imm = 0x4000000000000000
	.loc	1 27 11                         # dsp_op_delay_simple.py:27:11
	movq	%rax, (%r12)
	movabsq	$4616189618054758400, %rax      # imm = 0x4010000000000000
	.loc	1 28 11                         # dsp_op_delay_simple.py:28:11
	movq	%rax, (%r15)
	.loc	1 30 11                         # dsp_op_delay_simple.py:30:11
	cmpq	$1, %rbp
	jg	.LBB0_3
	.p2align	4, 0x90
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	movq	$0, (%r14,%rbp,8)
	incq	%rbp
	cmpq	$1, %rbp
	jle	.LBB0_2
.LBB0_3:
	.loc	1 0 11 is_stmt 0                # dsp_op_delay_simple.py:0:11
	xorl	%eax, %eax
	.loc	1 31 11 is_stmt 1               # dsp_op_delay_simple.py:31:11
	cmpq	$3, %rax
	jg	.LBB0_6
	.p2align	4, 0x90
.LBB0_5:                                # =>This Inner Loop Header: Depth=1
	movq	$0, (%rbx,%rax,8)
	incq	%rax
	cmpq	$3, %rax
	jle	.LBB0_5
.LBB0_6:
	.loc	1 0 11 is_stmt 0                # dsp_op_delay_simple.py:0:11
	xorl	%ebp, %ebp
	.loc	1 33 3 is_stmt 1                # dsp_op_delay_simple.py:33:3
	cmpq	$9, %rbp
	jg	.LBB0_9
	.p2align	4, 0x90
.LBB0_8:                                # =>This Inner Loop Header: Depth=1
	movsd	(%rbx,%rbp,8), %xmm0            # xmm0 = mem[0],zero
	movl	$frmt_spec, %edi
	movb	$1, %al
	callq	printf@PLT
	incq	%rbp
	cmpq	$9, %rbp
	jle	.LBB0_8
.LBB0_9:
	.loc	1 14 11                         # dsp_op_delay_simple.py:14:11
	movq	%r13, %rdi
	callq	free@PLT
	.loc	1 27 11                         # dsp_op_delay_simple.py:27:11
	movq	%r12, %rdi
	callq	free@PLT
	.loc	1 28 11                         # dsp_op_delay_simple.py:28:11
	movq	%r15, %rdi
	callq	free@PLT
	.loc	1 30 11                         # dsp_op_delay_simple.py:30:11
	movq	%r14, %rdi
	callq	free@PLT
	.loc	1 31 11                         # dsp_op_delay_simple.py:31:11
	movq	%rbx, %rdi
	callq	free@PLT
	.loc	1 10 1 epilogue_begin           # dsp_op_delay_simple.py:10:1
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Ltmp0:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	nl,@object                      # @nl
	.section	.rodata,"a",@progbits
nl:
	.asciz	"\n"
	.size	nl, 2

	.type	frmt_spec,@object               # @frmt_spec
frmt_spec:
	.asciz	"%f "
	.size	frmt_spec, 4

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	14                              # DW_FORM_strp
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	17                              # DW_AT_low_pc
	.byte	1                               # DW_FORM_addr
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	110                             # DW_AT_linkage_name
	.byte	14                              # DW_FORM_strp
	.byte	3                               # DW_AT_name
	.byte	14                              # DW_FORM_strp
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x39 DW_TAG_compile_unit
	.long	.Linfo_string0                  # DW_AT_producer
	.short	2                               # DW_AT_language
	.long	.Linfo_string1                  # DW_AT_name
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Linfo_string2                  # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	2                               # Abbrev [2] 0x2a:0x19 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string3                  # DW_AT_linkage_name
	.long	.Linfo_string3                  # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	10                              # DW_AT_decl_line
                                        # DW_AT_external
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"MLIR"                          # string offset=0
.Linfo_string1:
	.asciz	"dsp_op_delay_simple.py"        # string offset=5
.Linfo_string2:
	.asciz	"../mlir/test/Examples/Toy/Ch6" # string offset=28
.Linfo_string3:
	.asciz	"main"                          # string offset=58
	.section	.debug_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	68                              # Compilation Unit Length
	.long	42                              # DIE offset
	.asciz	"main"                          # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	68                              # Compilation Unit Length
	.long	0                               # End Mark
.LpubTypes_end0:
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
